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
import contextlib
import logging
import os
import time

import numpy as np

import tvm

import tvm.micro
import tvm.micro.device.arm.stm32f746xx as stm32f746xx

from micro_eval import model
from micro_eval.util import autotvm_log_util
from micro_eval.util import device_util
from micro_eval.util import log_util
from micro_eval.util import model_util

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


def get_num_devices(tracker_host, tracker_port, key):
    conn = tvm.rpc.connect_tracker(tracker_host, tracker_port)
    summary = conn.text_summary()
    num_connected = 0
    for line in summary.split('\n'):
        if 'Queue Status' in line:
            break
        if key in line:
            num_connected += 1
    return num_connected


def _tune_one_task(args, measure_option, task_index, num_tasks, autotvm_log_file, task):
    _LOG.info(f'starting task {task_index}: ({task.name}, {task.args})')
    prefix = "[Task %2d/%2d] " % (task_index, num_tasks)
    #tuner = tvm.autotvm.tuner.XGBTuner(task, loss_type='rank')
    tuner = tvm.autotvm.tuner.GATuner(task)

    n_trial = min(args.num_iterations, len(task.config_space))
    tuner.tune(n_trial=args.num_iterations,
               early_stopping=args.num_iterations,
               measure_option=measure_option,
               callbacks=[
                   tvm.autotvm.callback.progress_bar(args.num_iterations, prefix=prefix, si_prefix='k'),
                   tvm.autotvm.callback.log_to_file(autotvm_log_file)],
               si_prefix='k')


def tune_model(args, transport_launcher, model_inst):
    compiled_model = model_inst.build_model()
    tasks = model_inst.extract_tunable_tasks(compiled_model)
    _LOG.info(f'extracted {len(tasks)} tasks: {tasks}')
    for i, t in enumerate(tasks):
        _LOG.info(f' * Task {i:d}: config space is {len(t.config_space)}')

    if args.single_task_index:
        assert len(tasks) >= args.single_task_index, (
            f'--single-task-index={args.single_task_index}, but extracted only {len(tasks)} tasks')

        tasks = [tasks[args.single_task_index]]

    _LOG.info('[Tuning]')
    logging.getLogger('autotvm').setLevel(logging.INFO)

    # create tmp log file
    job_name = autotvm_log_util.compute_job_name(args.model_spec, model_inst)
    tuning_log = autotvm_log_util.gen_tuning_log_path(job_name)
    tmp_log_file = f'{tuning_log}.tmp'
    assert not os.path.exists(tmp_log_file)

    tmp_log_file_dir = os.path.dirname(tmp_log_file)
    if not os.path.isdir(tmp_log_file_dir):
        os.makedirs(tmp_log_file_dir)

    for i, task in enumerate(tasks):
        with contextlib.ExitStack() as exit_stack:
            section_constraints = model_inst.section_constraints(task_index_and_task=(i, task))
            if args.pre_launched_tracker_hostport:
                tracker_host, tracker_port = args.pre_launched_tracker_hostport.rsplit(':', 1)
                tracker_port = int(tracker_port)
                _LOG.warning('with pre-launched tracker, the micro.Session device config may become '
                             'out of sync with the device config used here to build models')
                target_num_servers = 1
            else:
                tracker_host, tracker_port = transport_launcher.tracker_host_port_tuple
                exit_stack.enter_context(transport_launcher.launch(
                    stm32f746xx.generate_config,
                    {'section_constraints': section_constraints}))
                target_num_servers = transport_launcher.num_instances

            num_servers = 0
            while num_servers < target_num_servers:
                num_servers = get_num_devices(tracker_host, tracker_port, transport_launcher.rpc_tracker_key)
                if num_servers < target_num_servers:
                    _LOG.info(
                        f'Found {num_servers} RPC servers under key {transport_launcher.rpc_tracker_key}, '
                        f'waiting for {target_num_servers} total to become available')
            _LOG.info(
                f'Discovered {num_servers} available RPC servers for key '
                f'{transport_launcher.rpc_tracker_key}')
            assert num_servers > 0, (
                f'No servers available on the tracker for key {transport_launcher.rpc_tracker_key}')

            dev_config = stm32f746xx.generate_config(
                tracker_host, tracker_port, section_constraints=section_constraints)

            measure_option = model_inst.get_autotvm_measure_option(
                num_servers, tracker_host, tracker_port, transport_launcher.rpc_tracker_key,
                dev_config, i, task)

            _tune_one_task(args, measure_option, i, len(tasks), tmp_log_file, task)

    return tasks, tmp_log_file


def analyze(args, model_inst, tasks, tmp_log_file, promote=False):
    _LOG.info('Best configs:')
    for i, task in enumerate(reversed(tasks)):
       # show best config from tuning
       dispatch_context = tvm.autotvm.apply_history_best(tmp_log_file)
       best_config = dispatch_context.query(task.target, task.workload)
       _LOG.info(f'  task.target: {task.target}')
       _LOG.info(f'  task {i}: {best_config}')

    # store best record in a cache file
    best_log_file = os.path.splitext(tmp_log_file)[0]
    tvm.autotvm.record.pick_best(tmp_log_file, best_log_file)
    _LOG.info(f'Wrote best configs to {best_log_file}')
    if promote:
        job_name = autotvm_log_util.compute_job_name(args.model_spec, model_inst)
        autotvm_log_util.promote(job_name, best_log_file)
        _LOG.info(f'Promoted {best_log_file} to the default tuning log for model spec {args.model_spec}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_spec',
                        help=('Specifies the model to tune in terms of model name, setting, '
                              'and config. Entries are of the form '
                              '<model_name>:[setting=]<setting>[:[config=]<config>]. <model_name> '
                              'is a string naming the Python module relative to micro_eval.models '
                              'that defines the TunableModule subclass to use. <setting> describes '
                              'the target and runtime used, and must be "micro_dev." <config> is '
                              'the path to a JSON file containing tweaks to the built module.'))
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
    parser.add_argument('--skip-writing-transport-config', action='store_true',
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

    launch_transport_parser = subparsers.add_parser('launch_transport')
    launch_transport_parser.add_argument(
        '--task-index', type=int, required=True,
        help=('If specified, the 0-based task index used to configure the TVM RPC server. If not '
              'specified, the TVM RPC server is configured for whole-model execution (which is not '
              'useful for autotuning, but would be useful for evaluation). At present, this mainly '
              "adjusts the TVM RPC server's section allocation map. This option may go away in the "
              'future should it become unnecessary to adjust this.'))


    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--log-file', required=True, help='path to log file to analyze')
    analyze_parser.add_argument('--promote', action='store_true', help='promote to symlinked default')

    rpc_server_dev_config_parser = subparsers.add_parser('rpc_dev_config')
    rpc_server_dev_config_parser.add_argument(
        '--task-index', type=int,
        help=('If specified, the 0-based task index used to configure the TVM RPC server. If not '
              'specified, the TVM RPC server is configured for whole-model execution (which is not '
              'useful for autotuning, but would be useful for evaluation). At present, this mainly '
              "adjusts the TVM RPC server's section allocation map. This option may go away in the "
              'future should it become unnecessary to adjust this.'))
    return parser.parse_args()


def _cmd_rpc_dev_config(args):
    log_util.config([], logging.INFO, console_only=True)
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    model_inst, _ = model.instantiate_from_spec(args.model_spec)
    index_and_task = None
    if args.task_index is not None:
        tasks = model_inst.extract_tunable_tasks(model_inst.build_model())
        index_and_task = (args.task_index, tasks[args.task_index])

    transport_launcher.generate_rpc_server_configs(
        tvm.micro.device.arm.stm32f746xx.generate_config,
        {'section_constraints': model_inst.section_constraints(index_and_task)})
    transport_launcher.generate_openocd_configs()
    print(f'Wrote OpenOCD and RPC server configs underneath {transport_launcher.work_dirtree_root}')

def _cmd_launch_transport(args):
    log_util.config(['autotune', args.model_spec])
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    generate_config = not args.skip_writing_transport_config

    launch_kw = {'generate_config': generate_config}
    if generate_config:
        model_inst, _ = model.instantiate_from_spec(args.model_spec)
        index_and_task = None
        if args.task_index is not None:
            tasks = model_inst.extract_tunable_tasks(model_inst.build_model())
            index_and_task = (args.task_index, tasks[args.task_index])
        launch_kw['generate_config_func'] = tvm.micro.device.arm.stm32f746xx.generate_config
        launch_kw['generate_config_kw'] = {
            'section_constraints': model_inst.section_constraints(index_and_task)}

    with transport_launcher.launch(**launch_kw):
        print('Transport launched. Press Ctrl+C to terminate.')
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print('Caught SIGINT; shutting down')

def _cmd_tune(args):
    transport_launcher = device_util.DeviceTransportLauncher({'use_tracker': True})
    log_util.config(['autotune', args.model_spec], logging.INFO)
    model_inst, _ = model.instantiate_from_spec(args.model_spec)
    tasks, log_file = tune_model(args, transport_launcher, model_inst)
    analyze(args, model_inst, tasks, log_file, promote=True)


def _cmd_analyze(args):
    log_util.config([], logging.INFO, console_only=True)
    model_inst, _ = model.instantiate_from_spec(args.model_spec)
    tasks = model_inst.extract_tunable_tasks(model_inst.build_model())
    analyze(args, model_inst, tasks, args.log_file, promote=args.promote)


def main():
    args = parse_args()
    globals()[f'_cmd_{args.action}'](args)


if __name__ == '__main__':
    main()
