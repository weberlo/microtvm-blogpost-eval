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
Auto-tuning convolution on ARM Cortex-M7 STM32F746 Boards
=====================================================================
**Author**: `Logan Weber <https://github.com/weberlo>`_

TODO More docs
"""
import logging
import os
import sys
from collections import OrderedDict

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
import tvm.micro.device.arm.stm32f746xx as stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint

from tvm.relay import transform
from tvm.relay.op import nn

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from micro_eval.util import gen_cifar10_cnn, relay_micro_build, custom_pick_best

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

####################
# Autotuning Setup #
####################
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
DEV_CONFIG['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(OrderedDict([
    ('text', (18000, MemConstraint.ABSOLUTE_BYTES)),
    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (100.0, MemConstraint.WEIGHT)),
    ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
]))

DEVICE_ID = 'arm.stm32f746xx'
TARGET = tvm.target.create('c -device=micro_dev')

#N_TRIAL = 1500
#EARLY_STOPPING = 800
N_TRIAL = 600
EARLY_STOPPING = 400
# we only need one per trial because the timings are cycle-accurate
N_PER_TRIAL = 15
# change this to the number of boards you have attached
#N_PARALLEL = 9
E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'

TRACKER_ADDR = '0.0.0.0'
TRACKER_PORT = 9190

#INPUT_SHAPE = (1, 3, 32, 32)

TUNE_OPS = [relay.op.nn.conv2d]

#N, H, W, CO, CI, KH, KW = 1, 32, 32, 3, 3, 5, 5
#STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
LAYOUT = 'NCHW'
#IN_DTYPE = 'float32'
#OUT_DTYPE = 'float32'
IN_DTYPE = 'int8'
OUT_DTYPE = 'int32'
# disable timeouts because JTAG is slow
TIMEOUT = 0
#assert N == 1, "Only consider batch_size = 1 in this template"

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


###################
# Autotuning/Eval #
###################

#def gen_conv2d_relay():
#    N, H, W, CO, CI = 1, 32, 32, 16, 3
#    KH, KW = 5, 5
#    STRIDES, PADDING, DILATION = (1, 1), (2, 2), (1, 1)
#    KERNEL_SIZE = (KH, KW)
#    DATA_SHAPE = (N, CI, H, W)
#    KERNEL_SHAPE = (CO, CI, KH, KW)
#    BIAS_SHAPE = (CO,)
#    OUTPUT_SHAPE = (N, H, W, CO)
#
#    #assert False, "we might need to use NCHW for micro and interp, because the bias add causes problems"
#    # Construct Relay program (used for micro and interpreter eval).
#    data_var = relay.var("data", shape=DATA_SHAPE, dtype=IN_DTYPE)
#    kernel_var = relay.var("kernel", shape=KERNEL_SHAPE, dtype=IN_DTYPE)
#    bias_var = relay.var("bias", shape=BIAS_SHAPE, dtype=OUT_DTYPE)
#    conv_expr = relay.nn.conv2d(
#            data_var, kernel_var,
#            kernel_size=KERNEL_SIZE,
#            strides=STRIDES,
#            padding=PADDING,
#            dilation=DILATION,
#            channels=CO,
#            data_layout=LAYOUT,
#            out_layout=LAYOUT,
#            out_dtype=OUT_DTYPE)
#    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
#    mod = relay.Module.from_expr(func)
#    mod = transform.InferType()(mod)
#    return mod

def get_num_devices(dev_id):
    conn = rpc.connect_tracker(TRACKER_ADDR, TRACKER_PORT)
    summary = conn.text_summary()
    num_connected = 0
    for line in summary.split('\n'):
        if 'Queue Status' in line:
            break
        if dev_id in line:
            num_connected += 1
    return num_connected


def tune_model(tasks):
    n_parallel = get_num_devices(DEVICE_ID)
    print('[Tuning]')
    for i in range(len(tasks)):
        if 'conv2d' in tasks[i].name:
            tasks[i] = autotvm.task.create(
                    tasks[i].name,
                    tasks[i].args,
                    tasks[i].target,
                    tasks[i].target_host,
                    template_key='direct')

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=tvm.micro.cross_compiler(DEV_CONFIG, micro.LibType.OPERATOR)),
            runner=autotvm.RPCRunner(DEVICE_ID, TRACKER_ADDR, TRACKER_PORT, n_parallel=n_parallel, number=N_PER_TRIAL, timeout=TIMEOUT)
            )

    # create tmp log file
    tmp_log_file = E2E_LOG_FILE_NAME + '.tmp'
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

    #print("\nBest configs:")
    #for i, task in enumerate(reversed(tasks)):
    #    # show best config from tuning
    #    dispatch_context = autotvm.apply_history_best(E2E_LOG_FILE_NAME)
    #    best_config = dispatch_context.query(task.target, task.workload)
    #    print(f'  task.target: {task.target}')
    #    print(f'  task {i}: {best_config}')

    # store best record in a cache file
    #autotvm.record.pick_best(tmp_log_file, E2E_LOG_FILE_NAME)
    custom_pick_best(tmp_log_file, E2E_LOG_FILE_NAME, top_k=5)
    os.remove(tmp_log_file)


def eval_model(mod, target):
    with micro.Session(DEV_CONFIG) as sess:
        graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, target)
        ctx = tvm.micro_dev(0)

        data_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=data_shape)).astype(IN_DTYPE), ctx)
        kernel_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
        kernel_tvm = tvm.nd.array(
            (np.random.uniform(size=kernel_shape)).astype(IN_DTYPE), ctx)

        graph_mod.set_input(key='data', value=data_tvm)
        graph_mod.set_input(key='kernel', value=kernel_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        # clear any previous batch times
        ctx.sync()
        sess.get_last_batch_time()
        results = []
        for _ in range(N_PER_TRIAL):
            graph_mod.run()
            ctx.sync()
            results.append(sess.get_last_batch_time())
        return np.mean(results), np.std(results)


def tune_and_eval_model():
    from tvm.autotvm.task.topi_integration import TaskExtractEnv
    #from mxnet.gluon.model_zoo.vision import get_model
    #block = get_model('mobilenetv2_0.25', pretrained=True)
    #mod, params = relay.frontend.from_mxnet(block, shape={'data': INPUT_SHAPE}, dtype=DTYPE)

    #mod = gen_conv2d_relay()
    #params = {}

    mod, params = gen_cifar10_cnn(use_random_params=False)

    tasks = autotvm.task.extract_from_program(mod["main"], target=TARGET,
            params=params, ops=TUNE_OPS)

    print(f'extracted {len(tasks)} tasks')
    assert tasks

    tune_model(tasks)

    #input('finished tuning...')

    #print('[[Evaluation]]')
    ## Load best schedules
    #print("[Tuned]")
    #with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
    #    with TARGET:
    #        tuned_mean_time, tuned_std_dev = eval_model(mod, TARGET)
    #        print("Mean inference time: %.2f ms (+/- %.2f ms)" % (tuned_mean_time, tuned_std_dev))

    #print("[Untuned]")
    #untuned_mean_time, untuned_std_dev = eval_model(mod, TARGET)
    #print("Mean inference time: %.2f ms (+/- %.2f ms)" % (untuned_mean_time, untuned_std_dev))
    #print("[MicroTVM Speedup]")
    #print(f"{untuned_mean_time / tuned_mean_time}")

    #assert False, "Task extraction is stateful and whichever eval is run first sets the schedule to be used on subsequent evals"


if __name__ == '__main__':
    tune_and_eval_model()
