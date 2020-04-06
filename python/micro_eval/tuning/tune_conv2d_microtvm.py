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
from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args

import tvm.micro as micro
from tvm.micro import create_micro_mod

from tvm.relay import transform
from tvm.relay.op import nn

from topi.util import get_const_tuple
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn.conv2d import conv2d, conv2d_nchw
from topi.generic import schedule_conv2d_nchw
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify, get_const_tuple, traverse_inline

from micro_eval.util import conv2d_arm_micro_nchw_template, reset_gdbinit

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

DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6668)

DEVICE_ID = 'arm.stm32f746xx'
TARGET = tvm.target.create('c -device=micro_dev')

#N_TRIAL = 1500
#EARLY_STOPPING = 800
N_TRIAL = 30
EARLY_STOPPING = 15
# we only need one per trial because the timings are cycle-accurate
N_PER_TRIAL = 15
# change this to the number of boards you have attached
N_PARALLEL = 10

# run more trials during eval to mitigate communication overhead and
# approximate the true speedup
N_EVAL_TRIALS = 20

TRACKER_ADDR = '0.0.0.0'
TRACKER_PORT = 9190

INPUT_SHAPE = (1, 3, 32, 32)

N, H, W, CO, CI, KH, KW = 1, 32, 32, 3, 3, 5, 5
STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
LAYOUT = 'NCHW'
DTYPE = 'float32'
DATA_TENSOR = ('TENSOR', (N, H, W, CI), DTYPE)
KERNEL_TENSOR = ('TENSOR', (CO, CI, KH, KW), DTYPE)
# disable timeouts because JTAG is slow
TIMEOUT = 0
assert N == 1, "Only consider batch_size = 1 in this template"

#############
# Debugging #
#############
reset_gdbinit()

########################
# Conv Autotuning/Eval #
########################
CONV2D_LOG_FILE_NAME = f'{DEVICE_ID}.conv2d.log'

def tune_conv(task):
    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=tvm.micro.cross_compiler(DEV_CONFIG, micro.LibType.OPERATOR)),
            runner=autotvm.RPCRunner(DEVICE_ID, TRACKER_ADDR, TRACKER_PORT, n_parallel=N_PARALLEL, number=N_PER_TRIAL, timeout=TIMEOUT)
            )

    # create tmp log file
    tmp_log_file = CONV2D_LOG_FILE_NAME + '.tmp'
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    tuner = XGBTuner(task, loss_type='rank')

    # start tuning
    tuner.tune(n_trial=min(N_TRIAL, len(task.config_space)),
            early_stopping=EARLY_STOPPING,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(N_TRIAL),
                autotvm.callback.log_to_file(tmp_log_file)])

    # store best record in a cache file
    autotvm.record.pick_best(tmp_log_file, CONV2D_LOG_FILE_NAME)
    os.remove(tmp_log_file)


def tune_and_eval_conv():
    print('[Tuning]')
    task = autotvm.task.create(conv2d_arm_micro_nchw_template,
            args=(DATA_TENSOR, KERNEL_TENSOR, STRIDES, PADDING, DILATION, LAYOUT, DTYPE),
            target=TARGET)

    tune_conv(task)
    # show best config from tuning
    dispatch_context = autotvm.apply_history_best(CONV2D_LOG_FILE_NAME)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    input('finished tuning...')

    print('[Evaluation]')
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    kernel_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)

    def eval_mod(c_mod):
        input(c_mod.get_source())
        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = create_micro_mod(c_mod, DEV_CONFIG)
            micro_func = micro_mod['conv2d']
            ctx = tvm.micro_dev(0)

            data_tvm = tvm.nd.array(data_np, ctx)
            kernel_tvm = tvm.nd.array(kernel_np, ctx)
            output_tvm = tvm.nd.empty([N, CO, H, W], ctx=ctx)
            ctx.sync()
            sess.get_last_batch_time()
            for _ in range(N_EVAL_TRIALS):
                micro_func(data_tvm, kernel_tvm, output_tvm)
            ctx.sync()
            elapsed_time = sess.get_last_batch_time()
            output_np = output_tvm.asnumpy()
        return elapsed_time, output_np

    # compile with default schedule
    with TARGET:
        sched, arg_bufs = conv2d_arm_micro_nchw_template(
                DATA_TENSOR, KERNEL_TENSOR, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
        c_mod = tvm.build(sched, arg_bufs, name='conv2d')
        untuned_time, untuned_output = eval_mod(c_mod)

    # compile kernels with history best records
    with autotvm.apply_history_best(CONV2D_LOG_FILE_NAME):
        with TARGET:
            sched, arg_bufs = conv2d_arm_micro_nchw_template(
                    DATA_TENSOR, KERNEL_TENSOR, STRIDES, PADDING, DILATION, LAYOUT, DTYPE)
            c_mod = tvm.build(sched, arg_bufs, name='conv2d')
            tuned_time, tuned_output = eval_mod(c_mod)

    print(f'  speedup: {untuned_time / tuned_time}')
    tvm.testing.assert_allclose(untuned_output, tuned_output, rtol=1e-6, atol=5e-5)


if __name__ == '__main__':
    tune_and_eval_conv()
