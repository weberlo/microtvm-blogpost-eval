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
from tvm.micro import create_micro_mod

from tvm.relay.op import nn
from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d_nchw

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

def schedule_conv2d_nchw(cfg, data, kernel, conv):
    s = tvm.create_schedule(conv.op)
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()

    assert False, "improve schedule. last tuning run didn't generate a speedup"

    cfg.define_split("tile_f", f, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    #cfg.define_split("tile_rc", rc, num_outputs=3)
    #cfg.define_split("tile_ry", ry, num_outputs=3)
    #cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    # tile reduction axes
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    rco, rcm, rci = cfg['tile_rc'].apply(s, conv, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, conv, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, conv, rx)
    s[conv].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    kernel_scope = n  # this is the scope to attach global config inside this kernel
    output = conv

    # tune unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    return s


@autotvm.template
def conv2d_nchw_template(N, H, W, CO, CI, KH, KW, strides, padding, dilation, layout, out_dtype):
    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')

    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)

    cfg = autotvm.get_config()
    sched = schedule_conv2d_nchw(cfg, data, kernel, conv)
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
N_PER_TRIAL = 9
# change this to the number of boards you have attached
N_PARALLEL = 10

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
    tasks = [autotvm.task.create(conv2d_nchw_template,
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
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    kernel_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)

    def eval_mod(c_mod):
        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
            micro_func = micro_mod['conv2d']
            ctx = tvm.micro_dev(0)

            data_tvm = tvm.nd.array(data_np, ctx)
            kernel_tvm = tvm.nd.array(kernel_np, ctx)
            output_tvm = tvm.nd.empty([N, CO, H, W], ctx=ctx)
            micro_func(data_tvm, kernel_tvm, output_tvm)
            output_np = output_tvm.asnumpy()
            elapsed_time = sess.get_last_batch_time()
        return elapsed_time, output_np

    # compile with default schedule
    with TARGET:
        sched, arg_bufs = conv2d_nchw_template(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
        c_mod = tvm.build(sched, arg_bufs, name='conv2d')
    untuned_time, untuned_output = eval_mod(c_mod)

    # compile kernels with history best records
    with autotvm.apply_history_best(LOG_FILE_NAME):
        with TARGET:
            sched, arg_bufs = conv2d_nchw_template(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
            c_mod = tvm.build(sched, arg_bufs, name='conv2d')
    tuned_time, tuned_output = eval_mod(c_mod)

    print(f'  speedup: {untuned_time / tuned_time}')
    tvm.testing.assert_allclose(untuned_output, tuned_output)


#########################
# Model Autotuning/Eval #
#########################
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
    tune_and_eval_conv()
    #tune_and_eval_model()
