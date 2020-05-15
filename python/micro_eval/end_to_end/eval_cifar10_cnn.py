import argparse
import collections
import datetime
import json
import os
import sys
import warnings

import colorama
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import graph_runtime, util
from tvm.contrib.debugger import debug_runtime, debug_result
from tvm import relay
from tvm import micro
from tvm import autotvm
from tvm.micro import create_micro_mod
from tvm.micro.device import MemConstraint
from tvm.micro.device import host
from tvm.micro.device.arm import stm32f746xx
from tvm.relay.testing import resnet
from topi.util import get_const_tuple

from micro_eval import util
from micro_eval.util import model_util
from micro_eval.model import cifar10_cnn


LOGGER = util.get_logger('cifar10_cnn.log')


def generate_config(openocd_server_hostport, section_constraints=None):
    addr, port = openocd_server_hostport.rsplit(':', 1)  # Support ipv6 :::6666 syntax.
    port = int(port)
    return stm32f746xx.generate_config(addr, port, section_constraints)


MICRO_HEADERS = util.CMSIS_HEADERS
MICRO_INCLUDE_PATHS = util.CMSIS_INCLUDE_PATHS


# CMSIS config
CIFAR10_SRC_PATH = f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn/cmsis_cifar10_cnn_tfl.c'
CIFAR10_INCLUDE_PATH = f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn'
CMSIS_SRC_PATHS = [
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_max_pool_s8.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c',
]


# CMSIS requires different section spacing than micro.
CMSIS_SEC_CONTRAINTS = {
    'text': (35000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (4096, MemConstraint.ABSOLUTE_BYTES),
    'data': (100000, MemConstraint.ABSOLUTE_BYTES),
    'bss': (1320, MemConstraint.ABSOLUTE_BYTES),
    'args': (4096, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (130000, MemConstraint.ABSOLUTE_BYTES),
    # NOTE we need a deeper stack: since they make deep func calls
    'stack': (1024, MemConstraint.ABSOLUTE_BYTES),
}


def eval_interp(args, target, samples):
    mod = model_util.build_relay_mod(args.cifar10_conv_op_impl, 'x86',
                                     use_random_params=not args.validate_against)
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        main_gv = relay.GlobalVar('main')
        ir_mod = tvm.IRModule({})
        ir_mod[main_gv] = mod.mod['main']
        intrp = relay.create_executor("debug", ir_mod)
        f = intrp.evaluate(main_gv)
        print('compiled', f)

    predictions = []
    for i, sample in enumerate(samples):
        data_nt = sample['data']
        if model_util.has_simd_strategy(args.cifar10_conv_op_impl):
            data_nt = data_nt.resize(data_nt.shape.as_template_for(C=4))

        data_np = data_nt.with_layout(mod.data_layout).data
        label = sample['label']
        result = f(data_np, *[mod.params[param.name_hint] for param in mod.mod['main'].params[1:]])
        predictions.append(result.asnumpy()[0])
        print('got prediction', predictions[-1])

    return predictions


def eval_cpu(args, target, samples):
    mod = model_util.build_relay_mod(args.cifar10_conv_op_impl, 'x86',
                                     use_random_params=not args.validate_against)
    with relay.build_config(opt_level=0, disabled_pass={"AlterOpLayout"}):
        graph, op_mod, params = relay.build(mod.mod['main'], target="llvm", params=mod.params)
        if util.DEBUG_MODE:
            graph_mod = debug_runtime.create(graph, op_mod, tvm.cpu(0), dump_root=f'{util.get_repo_root()}/debug/cpu')
        else:
            graph_mod = graph_runtime.create(graph, op_mod, tvm.cpu(0))

        graph_mod.set_input(**params)

    predictions = []
    for i, sample in enumerate(samples):
        data_nt = sample['data']
        if model_util.has_simd_strategy(args.cifar10_conv_op_impl):
            data_nt = data_nt.resize(data_nt.shape.as_template_for(C=4))
#        assert data_np.shape == DATA_SHAPE
#        assert data_np.dtype == IN_DTYPE
        data_tvm = tvm.nd.array(data_nt.data, ctx=tvm.cpu(0))
        print(data_nt.shape)
        graph_mod.set_input('data', data_tvm)
#        print('built', graph_mod.astext())
        graph_mod.run()
        predictions.append(graph_mod.get_output(0).asnumpy()[0])
        print('got prediction', predictions[-1])

    return predictions


def eval_cmsis(args, target, samples):
    cmsis_dev_config = generate_config(args.openocd_server_hostport, CMSIS_SEC_CONTRAINTS)
    util.reset_gdbinit(cmsis_dev_config)

    DATA_LAYOUT = 'NHWC'
    DATA_SHAPE = (1, 32, 32, 3)
    OUTPUT_SHAPE = (10,)

    IN_DTYPE = 'uint8'
    OUT_DTYPE = 'int8'

    # Begin a session
    LOGGER.debug('[Initting]')
    with micro.Session(cmsis_dev_config) as sess:
        # Build the function
        LOGGER.debug('[Building]')
        micro_mod = create_micro_mod(
            util.MockCMod([CIFAR10_SRC_PATH]),
            cmsis_dev_config,
            lib_src_paths=CMSIS_SRC_PATHS,
            lib_headers=util.CMSIS_HEADERS,
            lib_include_paths=util.CMSIS_INCLUDE_PATHS + [CIFAR10_INCLUDE_PATH])

        funcs = collections.OrderedDict([
            ('cifar10', util.LabelledShape(N=1, H=32, W=32, C=32, dtype='int8')),
#            ('conv2', util.LabelledShape(N=1, H=16, W=16, C=32, dtype='int8')),
#            ('conv3', util.LabelledShape(N=10, dtype='int8')),
#            ('arm_cifar10_cnn_wrapper', util.LabelledShape(N=1, X=10, dtype='int8')),
#            ('cifar10', util.LabelledShape(N=10, dtype='int8')),
        ])
        ctx = tvm.micro_dev(0)
        results = []
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            label = sample['label']

            data_np = data_nt.with_layout(DATA_LAYOUT).data
            assert data_np.shape == DATA_SHAPE
            assert data_np.dtype == IN_DTYPE
            last_output_tvm = tvm.nd.array(data_np, ctx=ctx)

            outputs = {}
            for func_name, output_shape in funcs.items():
                output_nt = output_shape.gen_zero_tensor()
                output_tvm = tvm.nd.array(output_nt.data, ctx=ctx)
                exec_time = util.benchmark_micro_func(sess, micro_mod[func_name], [last_output_tvm, output_tvm], 1)
                LOGGER.info(f'  {func_name} execution took {exec_time} milliseconds')
                outputs[func_name] = output_tvm.asnumpy()
                del last_output_tvm
                last_output_tvm = output_tvm

            with open(f'{util.get_repo_root()}/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params', 'wb') as params_f:
                params_f.write(debug_result.save_tensors(outputs))
            cmsis_output_np = output_tvm.asnumpy()

            # LOGGER.info(f'  output: {cmsis_output_np}')
            # prediction = cifar10_cnn.CIFAR10_CLASSES[np.argmax(cmsis_output_np)]
            # label = cifar10_cnn.CIFAR10_CLASSES[label]
            # LOGGER.info(f'  prediction was {prediction}')
            # LOGGER.info(f'  actual was {label}')

            results.append(cmsis_output_np)
            #results.append(np.array([0] * 10))

    return results


# Section constraints to use for the compiled uTVM CIFAR10 implementation.
MICRO_SEC_CONSTRAINTS = {
    'text': (23000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (300, MemConstraint.ABSOLUTE_BYTES),
    'data': (0x80, MemConstraint.ABSOLUTE_BYTES),
    'bss': (820, MemConstraint.ABSOLUTE_BYTES),
    'args': (4496, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (140000, MemConstraint.ABSOLUTE_BYTES),
    'stack': (128, MemConstraint.ABSOLUTE_BYTES),
}


def eval_utvm(args, target, samples):
    """Compile and evaluate uTVM CIFAR10 implementation.

    Params
    ------

    args:
    """
    micro_dev_config = generate_config(args.openocd_server_hostport, MICRO_SEC_CONSTRAINTS)
    util.reset_gdbinit(micro_dev_config)

    mod = model_util.build_relay_mod(args.cifar10_conv_op_impl, 'micro_dev',
                                     use_random_params=not args.validate_against)

    LOGGER.debug('[Initting]')
    with micro.Session(micro_dev_config) as sess:
        LOGGER.debug('[Building]')
        def build_graph_mod():
            return util.relay_micro_build(
                mod.mod['main'],
                micro_dev_config, target,
                params=mod.params,
                lib_headers=MICRO_HEADERS,
                lib_include_paths=MICRO_INCLUDE_PATHS)

        with target:
            if args.use_tuned_schedule:
                with autotvm.apply_history_best(args.use_tuned_schedule):
                    graph_mod = build_graph_mod()
            else:
                graph_mod = build_graph_mod()

        if args.use_tuned_schedule:
            LOGGER.info('[[Micro Tuned]]')
        else:
            LOGGER.info('[[Micro Untuned]]')

        predictions = []
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            if mod.has_simd_strategy:
                data_nt = data_nt.resize(data_nt.shape.as_template_for(C=4))
            data_np = data_nt.with_layout(mod.data_layout).data
            label = sample['label']

            assert data_np.shape == get_const_tuple(mod.mod['main'].params[0].checked_type.shape)

            # execute with `image` as the input
            LOGGER.debug('[Executing]')
            sess.get_last_batch_time()
            ctx = tvm.micro_dev(0)
            ctx.sync()
            graph_mod.run(data=data_np)
            ctx.sync()
            exec_time = sess.get_last_batch_time()
            LOGGER.info(f'  model execution took {exec_time} milliseconds')

            # get output
            micro_output_np = graph_mod.get_output(0).asnumpy()[0]
            LOGGER.info(f'  output: {micro_output_np}')
            # prediction = cifar10_cnn.CIFAR10_CLASSES[np.argmax(micro_output_np)]
            # label = cifar10_cnn.CIFAR10_CLASSES[label]
            # LOGGER.info(f'  prediction was {prediction}')
            # LOGGER.info(f'  actual was {label}')
            predictions.append(micro_output_np)

        return predictions


def load_outputs(path):
    with open(path, 'rb') as f:
        outputs = relay.load_param_dict(f.read())
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


ALL_MODELS = ['cmsis', 'utvm', 'interp', 'cpu']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', choices=ALL_MODELS, nargs='+', help='models to evaluate')
    parser.add_argument('--use-tuned-schedule', nargs='?', const='autotvm_logs/arm.stm32f746xx.e2e.log',
                        help=('Use a tuned schedule in evaluating the micro model. The path to the '
                              'tuned log can be given; if not, the default symlink generated by '
                              'tune_relay_microtvm.py is used.'))
    parser.add_argument('--num-samples', type=int, default=10, help='number of image samples to try')
    parser.add_argument('--openocd-server-hostport',
                        # NOTE: need to explicitly choose ipv4 address, not localhost.
                        default='127.0.0.1:6666',
                        help='Address of the OpenOCD TCL server to use for device communication.')
    parser.add_argument('--debug-runtime', action='store_true',
                        help='Use debug runtime and print graph debugging info.')
    parser.add_argument('--validate-against', choices=ALL_MODELS, const='cpu', nargs='?',
                        help='Validate on-device output against the given runtime (by default, cpu)')
    model_util.define_cifar10_conv_op_impl(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    util.DEBUG_MODE = args.debug_runtime
    LOGGER.info('Run: %r', sys.argv)

    target = tvm.target.create('c -device=micro_dev')

    samples = model_util.get_sample_points(args.num_samples)

    results = {}
    to_run = list(args.models)
    if args.validate_against and args.validate_against not in to_run:
        to_run.append(args.validate_against)
    for model_name in to_run:
        results[model_name] = globals()[f'eval_{model_name}'](args, target, samples)

    if args.validate_against:
        for i in range(args.num_samples):
            allclose = {}
            for model_name in args.models:
                allclose[model_name] = np.allclose(
                    results[model_name][i].astype('float32'),
                    results[args.validate_against][i].astype('float32'))

            print(f'Sample {i} ---->')
#            for model_name in args.models:
#                np.testing.assert_array_equal(results[model_name], results[args.validate_against])
#            print('all equal!')

            rows = []
            rows.append(['class ->', ''] + list(range(10))) #[str(x) for x in range(10)])
            for model_name in args.models:
                color = ''
                if model_name != args.validate_against:
                    if not allclose[model_name]:
                        color = f'{colorama.Fore.RED}'
                    else:
                        color = colorama.Fore.GREEN

                rows.append([model_name, color] + list(results[model_name][i]))
            rows.append([args.validate_against, ''] + results[args.validate_against][i].tolist())

            spacing = max(len(r[0]) for r in rows)
            format_string = f'{{0:{spacing}s}}'
            print('fmt', format_string)
            print(format_string.format(rows[0][0]) + ''.join(['{0:5d}'.format(c) for c in rows[0][2:]]))
            format_string = f'{{0:{spacing}s}}'
            for r in rows[1:]:
                print(r[1] + format_string.format(r[0]) + colorama.Style.RESET_ALL + ''.join([' {0:+04d}'.format(y) for y in r[2:]]))
            sys.exit(0)


    if args.debug_runtime:
        cpu_outputs = load_outputs(f'{util.get_repo_root()}/debug/cpu/_tvmdbg_ctx_CPU_0/output_tensors.params')
        save_outputs_json(cpu_outputs, 'cpu_output.json')

        micro_outputs = load_outputs(f'{util.get_repo_root()}/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
        save_outputs_json(micro_outputs, 'micro_output.json')
        sys.exit(0)

    if args.debug_runtime:
        micro_outputs = load_outputs(f'{util.MICRO_GDB_DEBUG_PATH}/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
        save_outputs_json(micro_outputs, 'micro_output.json')

        print(f'micro output keys: {micro_outputs.keys()}')
        for key in micro_outputs.keys():
            micro_val = micro_outputs[key]
            print()
            print()
            print(f'=================[{key}]===================')
            print(micro_val)
            input('========================================')

    if args.debug_runtime:
        cpu_outputs = load_outputs(f'{util.get_repo_root()}/debug/cpu/_tvmdbg_ctx_CPU_0/output_tensors.params')
        save_outputs_json(cpu_outputs, 'cpu_output.json')

        micro_outputs = load_outputs(f'{util.get_repo_root()}/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
        save_outputs_json(micro_outputs, 'micro_output.json')

        for key in cpu_outputs.keys():
            if key not in micro_outputs:
                print(f'{key} not found in micro output')
                print(f'cpu keys: {cpu_outputs.keys()}')
                print(f'micro keys: {micro_outputs.keys()}')
            cpu_val = cpu_outputs[key]
            micro_val = micro_outputs[key]
            if np.allclose(cpu_val.astype('float32'), micro_val.astype('float32')):
                input(f'{key} output matches!')
            else:
                print('========================================')
                print(cpu_val)
                print('----------------------------------------')
                print(micro_val)
                print('----------------------------------------')
                input(f'{key} output does not match!')
                print('========================================')
                print()
                print()


if __name__ == "__main__":
    main()
