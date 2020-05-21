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
"""Auto-tuning on ARM Cortex-M7 STM32F746 Boards."""
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

from micro_eval.util import device_util
from micro_eval.util import model_util
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

_LOG = logging.getLogger(__name__)


# disable timeouts because JTAG is slow
TIMEOUT_SEC = 0


def get_num_devices(tracker_host, tracker_port, key):
    conn = rpc.connect_tracker(tracker_host, tracker_port)
    summary = conn.text_summary()
    num_connected = 0
    for line in summary.split('\n'):
        if 'Queue Status' in line:
            break
        if dev_id in line:
            num_connected += 1
    return num_connected


def _tune_one_task(args, measure_option, task_index, num_tasks, autotvm_log_file, task):
    _LOG.info(f'starting task {i}: ({task.name}, {task.args})')
    prefix = "[Task %2d/%2d] " % (task_index, num_tasks)
    #tuner = XGBTuner(task, loss_type='rank')
    tuner = GATuner(task)

    n_trial = min(args.num_iterations, len(task.config_space))
    tuner.tune(n_trial=args.num_iterations,
               early_stopping=args.num_iterations,
               measure_option=measure_option,
               callbacks=[
                   autotvm.callback.progress_bar(args.num_iterations, prefix=prefix, si_prefix='k'),
                   autotvm.callback.log_to_file(autotvm_log_file)],
               si_prefix='k')


def tune_model(args, transport_launcher, model_inst):
    tasks = model_inst.get_tasks()
    _LOG.info(f'extracted {len(tasks)} tasks: {tasks}')
    for i, t in enumerate(tasks):
        _LOG.info(f' * Task {i:d}: config space is {len(t.config_space)}')

    return tasks
    if args.single_task_index:
        assert len(tasks) >= args.single_task_index, (
            f'--single-task-index={args.single_task_index}, but extracted only {len(tasks)} tasks')

        tasks = [tasks[args.single_task_index]]

    if args.pre_launched_tracker_hostport:
        tracker_host, tracker_port = args.pre_launched_tracker_hostport.rsplit(':', 1)
        tracker_port = int(tracker_port)
    else:
        tracker_host, tracker_port = transport_launcher.tracker_host_port_tuple

    num_servers = get_num_devices(tracker_host, tracker_port, transport_launcher.tracker_key)
    _LOG.info('Discovered {num_servers} available servers')
    assert num_servers > 0, (
        f'No servers available on the tracker for key {transport_launcher.tracker_key}')

    _LOG.info('[Tuning]')
    logging.getLogger('autotvm').setLevel(logging.INFO)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    # create tmp log file
    tuning_log = autotvm_log_util.gen_tuning_log_path(f'arm.stm32f746xx.{model_inst.get_config_str()}')
    tmp_log_flie = f'{tuning_log}.tmp'
    assert not os.path.exists(tmp_log_file)

    for i, task in enumerate(tasks):
        measure_option = model_inst.get_autotvm_measure_option(
            num_servers, tracker_host, tracker_port, transport_launcher.tracker_key, task)
        if args.pre_launched_tracker_hostport:
            _tune_one_task(args, measure_option, task_index, len(tasks), tmp_log_file, task)
        else:
            with transport_launcher.launch(
                    generate_config_func=micro.device.arm.stm32f746xx.gen_config,
                    generate_config_kw={
                        'section_constraints': model_inst.get_section_constraints(task)}):
                _tune_one_task(args, measure_option, task_index, len(tasks), tmp_log_file, task)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_specs',nargs='+',
                        help=('Specifies the models to evaluate in terms of model name, config, '
                              'and setting. Entries are of the form '
                              '<model_name>:[setting=]<setting>[:[config=]<config>]. <model_name> '
                              'is a string naming the Python module relative to micro_eval.models '
                              'that defines the TunableModule subclass to use. <setting> describes '
                              'the target and runtime used, and must be utvm if specified here. '
                              '<config> is the path to a JSON file containing tweaks to the '
                              'built module.'))
    parser.add_argument('--environment-config',
                        default=device_util.DeviceTransportLauncher.DEFAULT_ENVIRONMENT_CONFIG_PATH,
                        help='path to configuration file tree root')
    parser.add_argument('--num-devices', type=int, default=0,
                        help=('Limits the number of devices to control. If not specified, tries to '
                              'instantiate all devices listed in the environment config.'))
    parser.add_argument('--device-serial-numbers',
                        help=('If specified, tell OpenOCD to use connected devices with these specific '
                              'serial numbers (comma-separated). Must match the "hla_serial" key in '
                              'the environment config.'))
    parser.add_argument('--skip-writing-transport-config', type=bool, action='store_true',
                        help=("If specified, don't write transport (OpenOCD and TVM RPC Server) config."))
    subparsers = parser.add_subparsers(dest='action')

    tune_parser = subparsers.add_parser('tune')
    tune_parser.add_argument('--num-iterations', type=int, default=500, help='number of iterations to run')
    tune_parser.add_argument('--pre-launched-tracker-hostport',
                             help=("If specified, don't launch a tracker, RPC servers, or OpenOCDs. "
                                   "Instead, use the tracker at this hostport. This option can be "
                                   "used with the rpc_dev_config and launch_transport subcommands to "
                                   "tweak RPC server or OpenOCD configuration."))
    tune_parser.add_argument('--single-task-index', type=int,
                             help=('If specified, tune only the task with this index. This option can be '
                                   'used with rpc_dev_config and launch_transport subcommands to '
                                   'tweak RPC server or OpenOCD configuration.'))

    launch_transport = subparsers.add_parser('launch_transport')

    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--log-file', required=True, help='path to log file to analyze')
    analyze_parser.add_argument('--promote', action='store_true', help='promote to symlinked default')

    rpc_server_dev_config_parser = subparsers.add_parser('rpc_dev_config')
    return parser.parse_args()


def _cmd_rpc_dev_config(args):
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    model_inst = model.instantiate_from_args(args)
    transport_launcher.generate_rpc_server_configs(
        micro.device.arm.stm32f746xx.generate_config, model_inst.section_constraints())
    transport_launcher.generate_openocd_configs()
    print(f'Wrote OpenOCD and RPC server configs underneath {transport_launcher.work_dirtree_root}')

def _cmd_launch_transport(args):
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    generate_config = not args.skip_writing_transport_config

    launch_kw = {'generate_config': generate_config}
    if generate_config:
        model_inst = model.instantiate_from_args(args)
        launch_kw['generate_config_func'] = micro.device.arm.stm32f746xx.generate_config
        launch_kw['generate_config_kw'] = model_inst.section_constraints()

    with transport_launcher.launch(**kw):
        print('Transport launched. Press Ctrl+C to terminate.')
        try:
            time.sleep()
        except KeyboardInterrupt:
            print('Caught SIGINT; shutting down')

def _cmd_tune(args):
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    model_inst = model.instantiate_from_args(args)
    log_util.config(['autotune', model_inst.get_config_str()])
    target = tvm.target.create('c -device=micro_dev')
    tasks, log_file = tune_model(args, transport_launcher, model_inst)
    analyze(tasks, log_file, promote=True)


def _cmd_analyze(args):
    tasks = get_tasks(args)
    analyze(tasks, args.log_file, promote=args.promote)


def main():
    args = parse_args()
    globals()[f'_cmd_{args.action}'](args)


if __name__ == '__main__':
    main()
