import argparse
import collections
import contextlib
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import warnings

import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import graph_runtime
from tvm.contrib import util as contrib_util
import tvm.micro
from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device import MemConstraint


from micro_eval import dataset
from micro_eval import model
from micro_eval import util
from micro_eval.util import autotvm_log_util
from micro_eval.util import log_util
from micro_eval.util import model_util
from micro_eval.util import device_util
from micro_eval.model import cifar10_cnn


_LOG = logging.getLogger(__name__)


def adapt_all_inputs(model_inst, sample, ir_func):
    """Adapt graph inputs from the given sample.

    Params
    ------
    model_inst : TunableModel
        The TunableModel instance.
    sample : dataset.ModelParamsResults
        The sample.
    ir_func : tvm.relay.Function
        The IR function.

    Returns
    -------
    dict
    """
    inputs = model_inst.adapt_sample_inputs(sample.inputs)

    mod_inputs = {}
    for p in ir_func.params:
        if p.name_hint not in inputs:
            continue

        value = inputs[p.name_hint]
        if isinstance(value, util.LabelledTensor):
            value = value.data

        mod_inputs[p.name_hint] = value

    return mod_inputs


def adapt_all_outputs(model_inst, graph_mod, sample):
    assert graph_mod.get_num_outputs() == 1
    assert len(sample.outputs) == 1

    output_key = next(iter(sample.outputs.keys()))  # Right now we assume only 1 graph output.
    results = {output_key: graph_mod.get_output(0).asnumpy()}

    return model_inst.adapt_model_outputs(results)


def eval_interp(args, transport_launcher, model_inst, compiled_model, samples):
    assert hasattr(model_inst, 'interp_lower_config'), (
        'HACK: for now, models run under the interpreter need an attribute "interp_lower_config", '
        'which is tvm.transform.PassContext() used to evaluate the IRModule')

    with model_inst.interp_lower_config:
        main_gv = tvm.relay.GlobalVar(compiled_model.entry_point)
        ir_mod = tvm.IRModule({})
        ir_func = compiled_model.ir_mod[compiled_model.entry_point]
        ir_mod[main_gv] = ir_func
        intrp = tvm.relay.create_executor("debug", ir_mod)
        f = intrp.evaluate(main_gv)

    results = []
    for i, sample in enumerate(samples):
        inputs = adapt_all_inputs(model_inst, sample, compiled_model.ir_mod[compiled_model.entry_point])
        for p in ir_func.params:
            if p.name_hint not in inputs:
                inputs[p.name_hint] = compiled_model.params[p.name_hint]
        output = f(*[inputs[p.name_hint] for p in ir_func.params])

        output_key = next(iter(sample.outputs.keys()))  # Right now we assume only 1 graph output.
        outputs = {output_key: output.asnumpy()}
        result = model_inst.adapt_model_outputs(outputs)
        results.append(result)
        _LOG.info('got result: %r', results[-1])

    return results


def eval_cpu(args, transport_launcher, model_inst, compiled_model, samples):
    lowered = model_inst.lower_model(compiled_model)
    if args.use_debug_runtime:
        graph_mod = tvm.contrib.debugger.debug_runtime.create(
            lowered.graph, lowered.mod, tvm.cpu(0),
            dump_root=f'{util.get_repo_root()}/debug/cpu')
    else:
        graph_mod = tvm.contrib.graph_runtime.create(
            lowered.graph, lowered.mod, tvm.cpu(0))

    graph_mod.set_input(**lowered.params)

    results = []
    for i, sample in enumerate(samples):
        inputs = adapt_all_inputs(
            model_inst, sample,
            compiled_model.ir_mod[compiled_model.entry_point])
        for key in inputs:
            if key in sample.inputs:
                graph_mod.set_input(key, inputs[key])

        graph_mod.run()
        results.append(adapt_all_outputs(model_inst, graph_mod, sample))
        _LOG.info('got prediction: %r', results[-1])

    return results


def _launch_gdb():
    print('')
    print('Launch gdb in another terminal:')
    print(f'cd {util.get_repo_root()}/debug/micro && arm-none-eabi-gdb')
    print('')
    print('Press [Enter] when finished')
    input()


def eval_micro_dev(args, transport_launcher, model_inst, compiled_model, samples):
    if args.openocd_server_hostport:
        openocd_host, openocd_port = args.openocd_server_hostport.rsplit(':', 1)
        openocd_port = int(openocd_port)
    else:
        openocd_host, openocd_port = transport_launcher.openocd_host_port_tuple(0)

    config = stm32f746xx.generate_config(
            openocd_host, openocd_port, model_inst.section_constraints())
    if args.debug_micro_execution:
        config['debug_func'] = _launch_gdb

    util.reset_gdbinit(config)
    with tvm.micro.Session(config) as sess:
        _LOG.debug('[Executing]')
        lowered = model_inst.lower_model(compiled_model, dev_config=config)
        ctx = tvm.micro_dev(0)
        if args.use_debug_runtime:
            mod = tvm.contrib.debugger.debug_runtime.create(
                lowered.graph, lowered.mod, ctx,
                dump_root=f'{util.get_repo_root()}/debug/micro')
        else:
            mod = graph_runtime.create(lowered.graph, lowered.mod, ctx)

        mod.set_input(**lowered.params)

        results = []
        for i, sample in enumerate(samples):
            inputs = adapt_all_inputs(
                model_inst, sample,
                compiled_model.ir_mod[compiled_model.entry_point])

            mod.set_input(**inputs)
            ctx.sync()  # Ensure all args have been pushed to the device.
            sess.get_last_batch_time()  # Flush last batch time.
            mod.run()
            ctx.sync()   # This actually executes the on-device task queue.
            results.append(adapt_all_outputs(model_inst, mod, sample))
            exec_time_ms = sess.get_last_batch_time()
            _LOG.info('got prediction after %.3f ms: %r', exec_time_ms, results[-1])

    return results


def load_outputs(path):
    with open(path, 'rb') as f:
        outputs = tvm.relay.load_param_dict(f.read())
        res_outputs = {}
        for key, val in outputs.items():
            if '_0__' in key:
                key = key[:key.index('_0__')]
                if key.startswith('fused_nn_'):
                    key = key[len('fused_nn_'):]
                if key.endswith('_1'):
                    key = key[:-len('_1')]
                if key.endswith('_2'):
                    key = key[:-len('_2')]
            res_outputs[key] = val.asnumpy()
        return res_outputs


def save_outputs_json(outputs, path):
    import json
    res_outputs = {}
    for key, val in outputs.items():
        res_outputs[key] = val.tolist()
    with open(path, 'w') as f:
        json.dump(res_outputs, f, sort_keys=True, indent=4)


USE_DEFAULT_TUNED_SCHEDULE = '__use_default_tuned_schedule__'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_specs',nargs='+',
                        help=('Specifies the models to evaluate in terms of model name, setting, '
                              'and config. Entries are of the form '
                              '<model_name>:[setting=]<setting>[:[config=]<config>]. <model_name> '
                              'is a string naming the Python module relative to micro_eval.models '
                              'that defines the TunableModule subclass to use. <setting> describes '
                              'the target and runtime used, and is one of '
                              f'{{{",".join(model.SETTING_TO_TARGET_AND_CONTEXT)}}}'
                              '. <config> is the path to a JSON file containing tweaks to the '
                              'built module.'))
    parser.add_argument('--num-samples', type=int, default=10, help='number of image samples to try')
    parser.add_argument('--use-tuned-schedule',
                        nargs='?',
                        const=USE_DEFAULT_TUNED_SCHEDULE,
                        help=('Use a tuned schedule in evaluating the micro model. The path to the '
                              'tuned log can be given; if not, the default symlink generated by '
                              'micro_eval.bin.autotune is used.'))
    parser.add_argument('--environment-config',
                        default=device_util.DeviceTransportLauncher.DEFAULT_ENVIRONMENT_CONFIG_PATH,
                        help='path to configuration file tree root')
    parser.add_argument('--device-serial-number',
                        help=('If specified, tell OpenOCD to use the device with this specific '
                              'serial number. Must match the "hla_serial" key in the environment '
                              'config.'))
    parser.add_argument('--openocd-server-hostport',
                        help=('Address of the OpenOCD TCL server to use for device communication. '
                              'NOTE: you might need to choose an IPv4 address (i.e. 127.0.0.1) '
                              'here.'))
    parser.add_argument('--use-debug-runtime', action='store_true',
                        help=("Use debug runtime and print graph debugging info. This option is "
                              "useful when you're confident the program is executing correctly, but "
                              "you need to inspect output from each step in the graph."))
    parser.add_argument('--debug-micro-execution', action='store_true',
                        help=('When executing on micro_dev, launch GDB to complete execution. You '
                              'are expected to step the program through until the UTVMDone '
                              'breakpoint is reached.'))

    parser.add_argument('--validate-against', const='cpu', nargs='?',
                        help='Validate on-device output against the given runtime (by default, cpu)')
    model_util.define_cifar10_conv_op_impl(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    log_util.config(['eval', '-'.join(args.model_specs)])

    model_inst_setting = collections.OrderedDict()
    for spec in args.model_specs:
        assert spec not in model_inst_setting, f'spec {spec} given twice'
        model_inst_setting[spec] = model.instantiate_from_spec(spec)

    validate_against = None
    if args.validate_against:
        assert args.validate_against not in model_inst_setting, (
            f'--validate-against={args.validate_against} also given in model_specs '
            'command-line argument')
        validate_against = model.instantiate_from_spec(args.validate_against)
        model_inst_setting[args.validate_against] = validate_against

    dataset_generator_name = next(iter(model_inst_setting.values()))[0].dataset_generator_name()
    for spec, (m, _) in model_inst_setting.items():
        m.dataset_generator_name() == dataset_generator_name, (
            f'expected all models to share the same dataset, but {spec} has '
            f'{m.dataset_generator_name()}')
    dataset_gen = dataset.DatasetGenerator.instantiate(
        dataset_generator_name, {'shuffle': not validate_against})

    transport_launcher = None
    if (not args.openocd_server_hostport and
        any(setting == 'micro_dev' for _, setting in model_inst_setting.values())):
        run_options = {'use_tracker': False, 'num_instances': 1}
        if args.device_serial_number:
            run_options['hla_serials'] = [args.device_serial_number]
        transport_launcher = device_util.DeviceTransportLauncher(run_options)

    util.DEBUG_MODE = args.use_debug_runtime

    samples = dataset_gen.generate(args.num_samples)
    results = {}
    with contextlib.ExitStack() as all_models_stack:
        if args.debug_micro_execution:
            _LOG.warn('NOTE: to debug micro execution, compiled object files will be left in your '
                      'temp directory at: %s', contrib_util.TempDirectory._DEBUG_PARENT_DIR)
            _LOG.warn('This is a limitation of the current microTVM compilation structure')

            all_models_stack.enter_context(contrib_util.TempDirectory.set_keep_for_debug())

        for spec, (model_inst, setting) in model_inst_setting.items():
            with contextlib.ExitStack() as model_stack:
                if transport_launcher is not None:
                    model_stack.enter_context(transport_launcher.launch(
                        stm32f746xx.generate_config,
                        {'section_constraints': model_inst.section_constraints()}))

                if args.use_tuned_schedule:
                    if args.use_tuned_schedule == USE_DEFAULT_TUNED_SCHEDULE:
                        tuned_schedule = autotvm_log_util.get_default_log_path(
                            autotvm_log_util.compute_job_name(spec, model_inst))
                        if not os.path.exists(tuned_schedule):
                            _LOG.warning('Tuned schedule for %s not found; proceeding without: %s',
                                         spec, tuned_schedule)
                            tuned_schedule = None

                    else:
                        tuned_schedule = args.use_tuned_schedule

                    if tuned_schedule is not None:
                        model_stack.enter_context(autotvm.apply_history_best(tuned_schedule))

                compiled = model_inst.build_model()
                results[spec] = globals()[f'eval_{setting}'](
                    args, transport_launcher, model_inst, compiled, samples)

    if args.validate_against:
        for i in range(args.num_samples):
            allclose = {}
            for model_spec in args.model_specs:
                allclose[model_spec] = np.allclose(
                    results[model_spec][i]['label'].astype('float32'),
                    results[args.validate_against][i]['label'].astype('float32'))

            _LOG.info(f'Sample {i} ---->')
            rows = []
            rows.append([['model_name', 'setting', 'config']] + [x for x in range(10)])
            def _add_row(model_spec, values):
                model_spec_parts = model_spec.split(':', 3)
                if len(model_spec_parts) == 2:
                    model_spec_parts.append('')

                rows.append([model_spec_parts] + values)


            for model_spec in args.model_specs:
                color = ''
                if model_spec != args.validate_against:
                    if not allclose[model_spec]:
                        level = logging.ERROR
                    else:
                        level = logging.INFO
                    _add_row(model_spec, list(results[model_spec][i]['label']))

            _add_row(args.validate_against, results[args.validate_against][i]['label'].tolist())

            spacings = []
            for c in range(0, 3):
                spacing = max(len(r[0][c]) + 1 for r in rows)
                spacings.append(f'{{0:{spacing}s}}')

            _LOG.info(''.join([spacings[c].format(rows[0][0][c]) for c in range(0, 3)] +
                              ['{0:5d}'.format(c) for c in rows[0][1:]]))
            format_string = f'{{0:{spacing}s}}'
            for r in rows[1:]:
                model_spec_parts = ''.join([spacings[c].format(r[0][c]) for c in range(0, 3)])
                color = r[1]
                result_str = ''.join([' {0:+04d}'.format(y) for y in r[1:]])
                _LOG.log(level, '%s%s', model_spec_parts, result_str)


if __name__ == "__main__":
    main()
