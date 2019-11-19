# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning convolution on an ARM Cortex-M7 STM32F746 Board
=====================================================================
**Author**: `Logan Weber <https://github.com/weberlo>`_

TODO More docs
"""
import logging
import os
import sys

from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

import topi
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.micro as micro

from tvm.relay.op import nn
from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple

################
# Instructions #
################
#
# First, locate your OpenOCD script directory (e.g.,
# OPENOCD_SCRIPT_DIR=/usr/share/openocd/scripts) and run
#   `openocd -f $(OPENOCD_SCRIPT_DIR)/interface/stlink-v2-1.cfg -f $(OPENOCD_SCRIPT_DIR)/target/stm32f7x.cfg`
# in one terminal.
#
# If you want to connect multiple boards, you will need to
# identify the serial number of the JTAG adapter for each board.  To do so, use
# this trick:
# https://stackoverflow.com/questions/29121050/openocd-debugging-multiple-devices-at-once
#
# Once you have the serial numbers, create an OpenOCD `.cfg` file for each one,
# using the following template:
#   source [find target/stm32f7x.cfg]
#   hla_serial $SERIAL_NUMBER
#   gdb_port $GDB_PORT
#   tcl_port $TCL_PORT
#   telnet_port $TELNET_PORT
# Make sure that in each config file, the GDB, Tcl, and Telnet ports are unique
# across all config files.  We only care about the Tcl port, but OpenOCD will
# quit if *any* of the ports are already in use.
#
# With the config files created, use the following command, replacing
# $BOARD_CONFIG_FILE with each board's respective config file:
#   `openocd -f $(OPENOCD_SCRIPT_DIR)/interface/stlink-v2-1.cfg -f $BOARD_CONFIG_FILE
#
# Then, run
#   `python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port=9190`
# in another terminal.
#
# Then, run
#   `python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=micro --utvm-dev-id='arm.stm32f746xx' --utvm-dev-config-args='["127.0.0.1", 6666]'`
# in another terminal.  If you have multiple boards, you will need to run this
# command for each board, adjusting the port accordingly.
#


#########################################
# Compute/Schedule/Template Definitions #
#########################################

#
# The compute and schedule definitions have been adapted from the ARM CPU
# spatial pack templates to disable vectorization, as we do not currently emit
# vectorized instructions from the C codegen.
#
# The original definitions can be found here:
#   https://github.com/apache/incubator-tvm/blob/master/topi/python/topi/arm_cpu/conv2d_spatial_pack.py#L26
#

def conv2d_spatial_pack_nchw(cfg, data, kernel, strides, padding, dilation,
                             out_dtype, num_tile):
    """compute define for Conv2d Spatial Pack with NCHW layout"""
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, _, KH, KW, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    OH = (IH + pad_top + pad_bottom - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

    # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    if num_tile == 2:     # for arm cpu
        co, vc = cfg.define_split('tile_co', co, num_outputs=2)
        oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
        ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)
    elif num_tile == 3:   # for mali gpu
        co, _, vc = cfg.define_split('tile_co', co, num_outputs=3)
        oh, _, vh = cfg.define_split('tile_oh', oh, num_outputs=3)
        ow, _, vw = cfg.define_split('tile_ow', ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder("reorder_0",
                       [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                           [n, co, oh, ow, ci, kh, kw, vc, vh, vw]])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    #cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll')

    # fallback support
    if cfg.is_fallback:
        if num_tile == 2:     # arm cpu
            ref_log = autotvm.tophub.load_reference_log('arm_cpu', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
        elif num_tile == 3:  # mali gpu
            ref_log = autotvm.tophub.load_reference_log('mali', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, CI, KH, KW, VH, VW)
        data_vec = tvm.compute(dvshape, lambda n, h, w, ci, kh, kw, vh, vw:
                               data_pad[n][ci][(h*VH+vh)*HSTR+kh*dilation_h]
                               [(w*VW+vw)*WSTR+kw*dilation_w],
                               name='data_vec_undilated')
    else:
        dvshape = (N, OH // VH, OW // VW, CI, VH*HSTR + KH-1, VW*WSTR + KW-1)
        data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                               data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                               name='data_vec')

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(kvshape, lambda co, ci, kh, kw, vc:
                                 kernel[co*VC+vc][ci][kh][kw],
                                 name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    if dilation_h != 1 or dilation_w != 1:
        conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, ci, kh, kw, vh, vw].astype(out_dtype) *
                    kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')
    else:
        conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
                    kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n,
                              idxdiv(co, VC), idxdiv(h, VH), idxdiv(w, VW),
                              idxmod(h, VH), idxmod(w, VW), idxmod(co, VC)],
                         name='output_unpack', tag='spatial_conv2d_output')
    return output


def schedule_conv2d_spatial_pack_nchw(cfg, s, data_vec, kernel_vec,
                                      conv, output, last):
    """schedule implementation"""
    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[cfg['tile_oh'].size[-1],
                                        cfg['tile_ow'].size[-1],
                                        cfg['tile_co'].size[-1]],
                             max_unroll=None,
                             cfg=cfg)

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=None,
                                 cfg=cfg)
    s[conv].compute_at(s[last], ow)

    if data_vec.op.name == 'data_vec_undilated':
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis

    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _ = s[kernel_vec].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')

    return s


@autotvm.template
def conv2d_template(N, H, W, CO, CI, KH, KW, strides, padding, dilation, layout, out_dtype):
    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')

    cfg = autotvm.get_config()
    conv = conv2d_spatial_pack_nchw(
            cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    sched = schedule_conv2d_spatial_pack_nchw(cfg, [conv])
    return sched, [data, kernel, conv]


#####################
# Utility Functions #
#####################
def relay_micro_build(func, dev_config, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : TODO
        TODO

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device
    """
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target="c", params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


####################
# Autotuning Setup #
####################
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
DEV_CREATE_MICRO_LIB = micro.device.get_device_funcs(DEV_CONFIG['device_id'])['create_micro_lib']

DEVICE_ID = 'arm.stm32f746xx'
TARGET = tvm.target.create('c -device=micro_dev')

#N_TRIAL = 1500
#EARLY_STOPPING = 800
N_TRIAL = 300
EARLY_STOPPING = 150
# we only need one per trial because the timings are cycle-accurate
N_PER_TRIAL = 3
# change this to the number of boards you have attached
N_PARALLEL = 8

TRACKER_ADDR = '0.0.0.0'
TRACKER_PORT = 9190

LOG_FILE_NAME = f'{DEVICE_ID}.log'

INPUT_SHAPE = (1, 3, 32, 32)

N, H, W, CO, CI, KH, KW = 1, 16, 16, 3, 3, 5, 5
STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
LAYOUT = 'NCHW'
DTYPE = 'float32'
# disable timeouts because JTAG is slow
TIMEOUT = 0
assert N == 1, "Only consider batch_size = 1 in this template"

#############
# Debugging #
#############
def reset_gdbinit():
    if 'server_port' not in DEV_CONFIG:
        return
    with open('/home/lweber/gdb-conf/.gdbinit', 'w') as f:
        gdb_port = DEV_CONFIG['server_port'] - 3333
        gdbinit_contents = (
f"""layout asm
target remote localhost:{gdb_port}
set $pc = UTVMInit
break UTVMDone
""")
        f.write(gdbinit_contents)
reset_gdbinit()

########################
# Conv Autotuning/Eval #
########################
def tune_and_eval_conv():
    print('[Tuning]')
    tasks = [autotvm.task.create(conv2d_template,
            args=(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, DTYPE),
            target=TARGET)]

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=tvm.micro.cross_compiler(DEV_CREATE_MICRO_LIB, DEV_CONFIG['mem_layout'], micro.LibType.OPERATOR)),
            runner=autotvm.RPCRunner(DEVICE_ID, TRACKER_ADDR, TRACKER_PORT, n_parallel=N_PARALLEL, number=N_PER_TRIAL, timeout=TIMEOUT)
            )

    # create tmp log file
    tmp_log_file = LOG_FILE_NAME + '.tmp'
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
        tuner = XGBTuner(task, loss_type='rank')

        # start tuning
        tuner.tune(n_trial=min(N_TRIAL, len(task.config_space)),
                early_stopping=EARLY_STOPPING,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(N_TRIAL, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file)])

    # store best record in a cache file
    autotvm.record.pick_best(tmp_log_file, LOG_FILE_NAME)
    os.remove(tmp_log_file)

    input('finished tuning...')
    print('[Evaluation]')

    def eval_mod(c_mod):
        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = sess.create_micro_mod(c_mod)
            micro_func = micro_mod['conv2d']
            ctx = tvm.micro_dev(0)

            data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
            data_tvm = tvm.nd.array(data_np, ctx)
            kernel_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
            kernel_tvm = tvm.nd.array(kernel_np, ctx)
            c_tvm = tvm.nd.empty([N, CO, H, W], ctx=ctx)
            res = micro_func(data_tvm, kernel_tvm, c_tvm)
        return res

    # compile with default schedule
    with TARGET:
        sched, arg_bufs = conv2d_template(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
        c_mod = tvm.build(sched, arg_bufs, name='conv2d')
    default_cycle_count = eval_mod(c_mod)

    # compile kernels with history best records
    with autotvm.apply_history_best(LOG_FILE_NAME):
        with TARGET:
            sched, arg_bufs = conv2d_template(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
            c_mod = tvm.build(sched, arg_bufs, name='conv2d')
    autotune_cycle_count = eval_mod(c_mod)

    print(f'  speedup: {default_cycle_count / autotune_cycle_count}')


#########################
# Model Autotuning/Eval #
#########################
def get_micro_model():
    pass


def tune_and_eval_model():
    print('[Tuning]')
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('mobilenetv2_0.25', pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={'data': INPUT_SHAPE}, dtype=DTYPE)
    print(mod)
    input('ayy')

    def tune_model():
        tasks = autotvm.task.extract_from_program(mod["main"], target=TARGET,
                params=params, ops=(relay.op.nn.conv2d,))

        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(
                    build_func=tvm.micro.cross_compiler(DEV_CREATE_MICRO_LIB, DEV_CONFIG['mem_layout'], micro.LibType.OPERATOR)),
                runner=autotvm.RPCRunner(DEVICE_ID, TRACKER_ADDR, TRACKER_PORT, n_parallel=N_PARALLEL, number=N_PER_TRIAL, timeout=TIMEOUT)
                )

        # create tmp log file
        tmp_log_file = LOG_FILE_NAME + '.tmp'
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, task in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
            tuner = XGBTuner(task, loss_type='rank')

            # start tuning
            tuner.tune(n_trial=min(N_TRIAL, len(task.config_space)),
                    early_stopping=EARLY_STOPPING,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(N_TRIAL, prefix=prefix),
                        autotvm.callback.log_to_file(tmp_log_file)])

        # store best record in a cache file
        autotvm.record.pick_best(tmp_log_file, LOG_FILE_NAME)
        os.remove(tmp_log_file)

    def eval_model():
        print('[Evaluation]')
        graph_mod = relay_micro_build(mod['main'], DEV_CONFIG)

        data_tvm = tvm.nd.array(
            (np.random.uniform(size=input_shape)).astype(input_dtype), ctx)
        kernel_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
        kernel_tvm = tvm.nd.array(
            (np.random.uniform(size=kernel_shape)).astype(input_dtype), ctx)

        graph_mod.set_input(**params)
        graph_mod.set_input(key='data', value=data_tvm)
        graph_mod.set_input(key='kernel', value=kernel_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = graph_mod.module.time_evaluator("run", ctx, number=5, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

    #tune_model()
    #input('finished tuning...')
    with micro.Session(DEV_CONFIG):
        # Load best schedules
        with autotvm.tophub.context(TARGET, extra_files=[LOG_FILE_NAME]):
            eval_model()


if __name__ == '__main__':
    #tune_and_eval_conv()
    tune_and_eval_model()
