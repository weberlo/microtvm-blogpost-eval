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
import argparse
import datetime
import json
import logging
import os
import os.path
import signal
import subprocess
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

from micro_eval.util import (
    CMSIS_NN_PATH, CMSIS_HEADERS, CMSIS_INCLUDE_PATHS,
    NamedTensor, NamedType, BakedType,
    print_c_source,
    custom_pick_best,
    relay_micro_build, reset_gdbinit,
    get_comm_overhead, benchmark_micro_func,
    check_conv2d_output
)
from micro_eval.model.cifar10_cnn import gen_cifar10_cnn
from micro_eval.micro_topi import collect_conv_tasks
from micro_eval.micro_topi.cortex_m7.conv2d.direct import conv2d_direct
from micro_eval.micro_topi.cortex_m7.conv2d.direct_simd import conv2d_direct_simd
from micro_eval.micro_topi.cortex_m7.conv2d.partial_im2col import conv2d_partial_im2col

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
logging.getLogger('autotvm').setLevel(logging.INFO)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

OBJ_BUILD_CONFIG = micro.device.arm.stm32f746xx.generate_config('127.0.0.1', 6666, section_constraints={
    'text': (50000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (100, MemConstraint.ABSOLUTE_BYTES),
    'data': (100, MemConstraint.ABSOLUTE_BYTES),
    'bss': (600, MemConstraint.ABSOLUTE_BYTES),
    'args': (4096, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (100000, MemConstraint.ABSOLUTE_BYTES),
    'stack': (32, MemConstraint.ABSOLUTE_BYTES),
})

DEVICE_ID = 'arm.stm32f746xx'
TARGET = tvm.target.create('c -device=micro_dev')

# N_TRIAL = 1500
# EARLY_STOPPING = 800
# N_TRIAL = 250
# EARLY_STOPPING = 250
# N_TRIAL = 30
# EARLY_STOPPING = 30
# N_TRIAL = 1
# EARLY_STOPPING = 1

TRACKER_ADDR = '0.0.0.0'
TRACKER_PORT = 9190

TUNE_OPS = [relay.op.nn.conv2d]

# disable timeouts because JTAG is slow
TIMEOUT = 0

#############
# Debugging #
#############
# NOTE in the autotvm setting, this is only useful if there's only one RPC server running
# reset_gdbinit(DEV_CONFIG)

###################
# Autotuning/Eval #
###################

#def gen_conv2d_relay():
#    IN_DTYPE = 'int8'
#    OUT_DTYPE = 'int32'
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


def tune_model(rpc_config_base, num_servers, num_trials, tasks, log_file_name):
    print('[Tuning]')

    builder = autotvm.LocalBuilder(
        build_func=tvm.micro.cross_compiler(OBJ_BUILD_CONFIG, micro.LibType.OPERATOR, lib_headers=CMSIS_HEADERS, lib_include_paths=CMSIS_INCLUDE_PATHS),
        n_parallel=num_servers)
    builder.build_kwargs.setdefault('build_option', {})['disable_vectorize'] = True
    runner = autotvm.RPCRunner(DEVICE_ID, TRACKER_ADDR, TRACKER_PORT, n_parallel=num_servers, number=1, repeat=1, timeout=TIMEOUT)

    measure_option = autotvm.measure_option(builder=builder, runner=runner)

    # create tmp log file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_file_name_timestamp = f'{log_file_name}.{timestamp}'
    tmp_log_file = f'{log_file_name_timestamp}.tmp'
    assert not os.path.exists(tmp_log_file)

    for i, task in enumerate(tasks):
        update_rpc_server_config(rpc_config_base, num_servers, i, task)
        servers = launch_rpc_servers(rpc_config_base, num_servers)
        try:
            #input(f'starting task {i}: ({task.name}, {task.args})')
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
            #tuner = XGBTuner(task, loss_type='rank')
            tuner = GATuner(task)

            # start tuning
            n_trial = min(num_trials, len(task.config_space))
            tuner.tune(n_trial=n_trial,
                       early_stopping=n_trial, #EARLY_STOPPING,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix, si_prefix='k'),
                           autotvm.callback.log_to_file(tmp_log_file)],
                       si_prefix='k')
        finally:
            stop_rpc_servers(servers)

    return tmp_log_file


def analyze(tasks, tmp_log_file, promote=False):
    print("\nBest configs:")
    for i, task in enumerate(reversed(tasks)):
       # show best config from tuning
       dispatch_context = autotvm.apply_history_best(tmp_log_file)
       best_config = dispatch_context.query(task.target, task.workload)
       print(f'  task.target: {task.target}')
       print(f'  task {i}: {best_config}')

    # store best record in a cache file
    best_log_file = os.path.splitext(tmp_log_file)[0]
    autotvm.record.pick_best(tmp_log_file, best_log_file)
    if promote:
        symlink_path, timestamp = os.path.splitext(best_log_file)
        if os.path.lexists(symlink_path):
            if not os.path.islink(symlink_path):
                os.rename(symlink_path, f'{symlink_path}.moved-aside-for-{timestamp}')
            else:
                os.unlink(symlink_path)

        os.symlink(os.path.basename(tmp_log_file), symlink_path)


def compute_rpc_server_config_file_name(dev_config_base, i):
    return f'{dev_config_base}/{i}/utvm_dev_config.json'


WORKSPACE_SIZE_BYTES_BY_TASK_INDEX = [132000, 132000, 10000]


def update_rpc_server_config(config_base, num_servers, task_index, task):
    """Update RPC server config with task-specific config."""
    for i in range(num_servers):
        with open(compute_rpc_server_config_file_name(config_base, i)) as f:
            config = json.load(f)

        if task.name == 'conv2d_direct.arm_cpu':
            config['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(
                micro.device.arm.stm32f746xx.BASE_ADDR,
                micro.device.arm.stm32f746xx.AVAILABLE_MEM,
                micro.device.arm.stm32f746xx.WORD_SIZE,
                OrderedDict([
                    ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
                    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
                    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                    ('heap', (100.0, MemConstraint.WEIGHT)),
                    ('workspace', (WORKSPACE_SIZE_BYTES_BY_TASK_INDEX[task_index], MemConstraint.ABSOLUTE_BYTES)),
                    ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
                ]))
        elif task.name == 'conv2d_direct_simd.arm_cpu':
            config['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(
                micro.device.arm.stm32f746xx.BASE_ADDR,
                micro.device.arm.stm32f746xx.AVAILABLE_MEM,
                micro.device.arm.stm32f746xx.WORD_SIZE,
                OrderedDict([
                    ('text', (23000, MemConstraint.ABSOLUTE_BYTES)),
                    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
                    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                    ('heap', (100.0, MemConstraint.WEIGHT)),
                    ('workspace', (WORKSPACE_SIZE_BYTES_BY_TASK_INDEX[task_index], MemConstraint.ABSOLUTE_BYTES)),
                    ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
                ]))
        else:
            return

        print(f'updated config for server {i}')
        with open(compute_rpc_server_config_file_name(config_base, i), 'w') as f:
            json.dump(config, f, indent=4)


DEBUG_RPC_SERVER_OUTPUT = False


def launch_rpc_servers(config_base, num_servers):
    procs = []
    for i in range(num_servers):
        config = compute_rpc_server_config_file_name(config_base, i)
        procs.append(subprocess.Popen(
            [sys.executable, '-m', 'tvm.exec.rpc_server',
             '--tracker=0.0.0.0:9190',
             '--key=arm.stm32f746xx',
             '--utvm-dev-config={}'.format(config)],
            stdout=None if DEBUG_RPC_SERVER_OUTPUT else subprocess.DEVNULL,
            stderr=None if DEBUG_RPC_SERVER_OUTPUT else subprocess.DEVNULL))
    return procs


def stop_rpc_servers(procs):
    for p in procs:
        p.send_signal(signal.SIGINT)

    for p in procs:
        p.wait()


# def eval_model(mod, target):
#     with micro.Session(DEV_CONFIG) as sess:
#         graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, target)
#         ctx = tvm.micro_dev(0)

#         data_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
#         data_tvm = tvm.nd.array(
#             (np.random.uniform(size=data_shape)).astype(IN_DTYPE), ctx)
#         kernel_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
#         kernel_tvm = tvm.nd.array(
#             (np.random.uniform(size=kernel_shape)).astype(IN_DTYPE), ctx)

#         graph_mod.set_input(key='data', value=data_tvm)
#         graph_mod.set_input(key='kernel', value=kernel_tvm)

#         # evaluate
#         print("Evaluate inference time cost...")
#         # clear any previous batch times
#         ctx.sync()
#         sess.get_last_batch_time()
#         results = []
#         for _ in range(N_PER_TRIAL):
#             graph_mod.run()
#             ctx.sync()
#             results.append(sess.get_last_batch_time())
#         return np.mean(results), np.std(results)


def write_rpc_server_config(template_key, config_base, num_ports):
    # each op strategy needs a slightly different memory layout, so we update
    # the dev config the RPC servers use (only works if the script that restarts the RPC
    # server upon file modification is used)
    if template_key == 'direct':
        DEV_CONFIG['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(
            micro.device.arm.stm32f746xx.BASE_ADDR,
            micro.device.arm.stm32f746xx.AVAILABLE_MEM,
            micro.device.arm.stm32f746xx.WORD_SIZE,
            OrderedDict([
                ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ]))
    elif template_key == 'direct_simd':
        DEV_CONFIG['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(
            micro.device.arm.stm32f746xx.BASE_ADDR,
            micro.device.arm.stm32f746xx.AVAILABLE_MEM,
            micro.device.arm.stm32f746xx.WORD_SIZE,
            OrderedDict([
                ('text', (23000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (10000, MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ]))
    elif template_key == 'partial_im2col':
        DEV_CONFIG['mem_layout'] = micro.device.arm.stm32f746xx.gen_mem_layout(
            micro.device.arm.stm32f746xx.BASE_ADDR,
            micro.device.arm.stm32f746xx.AVAILABLE_MEM,
            micro.device.arm.stm32f746xx.WORD_SIZE,
            OrderedDict([
                ('text', (18000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
                # ('workspace', (64000, MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ]))
    else:
        assert False

    for i in range(num_ports):
        DEV_CONFIG['server_port'] = 6666 + i
        with open(f'{dev_config_base}/{i}/utvm_dev_config.json', 'w') as f:
            json.dump(DEV_CONFIG, f, indent=4)


def get_tasks(template_key):
    from tvm.autotvm.task.topi_integration import TaskExtractEnv
    TaskExtractEnv()

    # if template_key == 'direct':
    #     @autotvm.task.register('topi_nn_conv2d', override=True)
    #     def _conv2d_direct(*args, **kwargs):
    #         return conv2d_direct(*args, **kwargs)
    #     data_layout = conv2d_direct.default_data_layout
    #     kernel_layout = conv2d_direct.default_kernel_layout
    # elif template_key == 'direct_simd':
    #     @autotvm.task.register('topi_nn_conv2d', override=True)
    #     def _conv2d_direct_simd(*args, **kwargs):
    #         return conv2d_direct_simd(*args, **kwargs)
    #     data_layout = conv2d_direct_simd.default_data_layout
    #     kernel_layout = conv2d_direct_simd.default_kernel_layout
    # elif template_key == 'partial_im2col':
    #     @autotvm.task.register('topi_nn_conv2d', override=True)
    #     def _conv2d_partial_im2col(*args, **kwargs):
    #         return conv2d_partial_im2col(*args, **kwargs)
    #     data_layout = conv2d_partial_im2col.default_data_layout
    #     kernel_layout = conv2d_partial_im2col.default_kernel_layout
    # else:
    #     assert False

    #from mxnet.gluon.model_zoo.vision import get_model
    #block = get_model('mobilenetv2_0.25', pretrained=True)
    #mod, params = relay.frontend.from_mxnet(block, shape={'data': INPUT_SHAPE}, dtype=DTYPE)

    #mod, params = gen_conv2d('NHWC', 'HWOI')

    data_layout = 'NHWC'
    kernel_layout = 'HWOI'
    mod, params = gen_cifar10_cnn(
        data_layout, kernel_layout, op_strategy=template_key, use_random_params=True)


    with tvm.target.build_config(opt_level=3, disable_vectorize=True):
        tasks = autotvm.task.extract_from_program(mod['main'], params, TARGET)

#    tasks = collect_conv_tasks(mod['main'], TARGET, template_key)

    # dumb_tasks = autotvm.task.extract_from_program(
    #     mod['main'], target=TARGET, params=params, ops=TUNE_OPS)
    print(f'extracted {len(tasks)} tasks: {tasks}')
    assert len(tasks) == 3

    # for i in range(len(tasks)):
    #     assert 'conv2d' in tasks[i].name
    #     # overwrite template key (defaults to 'direct') with the desired key
    #     tasks[i] = autotvm.task.create(
    #             tasks[i].name,
    #             tasks[i].args,
    #             tasks[i].target,
    #             tasks[i].target_host,
    #             template_key=template_key)

    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-simd', action='store_true', help='Use direct_simd ops')
    subparsers = parser.add_subparsers(dest='action')

    tune_parser = subparsers.add_parser('tune')
    tune_parser.add_argument('--num-rpc-servers', type=int, default=1, help='number of rpc servers to launch, equal to number of boards')
    tune_parser.add_argument('--rpc-server-config-base', default='microrpc-dev-config', help='path to configuration file tree root')
    tune_parser.add_argument('--num-trials', type=int, default=500, help='number of trials to run')

    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--log-file', required=True, help='path to log file to analyze')
    analyze_parser.add_argument('--promote', action='store_true', help='promote to symlinked default')

    rpc_server_dev_config_parser = subparsers.add_parser('rpc_dev_config')
    rpc_server_dev_config_parser.add_argument('config_base', help='path to configuration file tree root')
    rpc_server_dev_config_parser.add_argument('--num-rpc-servers', default=10, type=int, help='number of dev server config files to write')

    return parser.parse_args()


def _build_template_keys(args):
    return ['direct_simd'] if args.use_simd else ['direct']


def _cmd_rpc_dev_config(args):
    for key in _build_template_keys(args):
        write_rpc_server_config(key, args.config_base, args.num_ports)


def _cmd_tune(args):
    template_keys = _build_template_keys(args)
    tasks = get_tasks(template_keys[0])
    log_file_name = f'{DEVICE_ID}.{template_keys[0]}.e2e.log'
    log_file = tune_model(args.rpc_server_config_base, args.num_rpc_servers, args.num_trials, tasks, log_file_name)
    analyze(tasks, log_file, promote=True)

def _cmd_analyze(args):
    template_keys = _build_template_keys(args)
    tasks = get_tasks(template_keys[0])
    analyze(tasks, args.log_file, promote=args.promote)

def main():
    args = parse_args()
    globals()[f'_cmd_{args.action}'](args)
#    assert False, 'make it so you\'re using all 3 tasks again'


if __name__ == '__main__':
    main()
    #assert False, "Task extraction is stateful and whichever eval is run first sets the schedule to be used on subsequent evals"
