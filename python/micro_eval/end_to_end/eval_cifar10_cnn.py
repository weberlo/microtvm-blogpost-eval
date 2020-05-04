import argparse
import collections
import datetime
import json
import os
import sys
import warnings

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import graph_runtime, util
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


# Model/Data util
def get_sample_points(n):
    """Grabs a single input/label pair from MNIST"""
    ctx = mx.cpu()
    # Load a random image from the test dataset
    sample_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=False),
            1,
            shuffle=True)

    samples = []
    for i, (data, label) in zip(range(n), sample_data):
        if i == n:
            break
        data_np = data.as_in_context(ctx).asnumpy()
        # gluon data is in NHWC format
        data_nt = util.NamedTensor(data_np, 'NHWC')
        label = int(label.asnumpy()[0])
        samples.append({'data': data_nt, 'label': label})
    return samples


# CMSIS config
CIFAR10_SRC_PATH = f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn/cmsis_cifar10_cnn.c'
CIFAR10_INCLUDE_PATH = f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn'
CMSIS_SRC_PATHS = [
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c',
]

# Evaluation
CIFAR10_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
NUM_SAMPLES = 10


# CMSIS requires different section spacing than micro.
CMSIS_SEC_CONTRAINTS = {
    'text': (25000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (4096, MemConstraint.ABSOLUTE_BYTES),
    'data': (100000, MemConstraint.ABSOLUTE_BYTES),
    'bss': (1024, MemConstraint.ABSOLUTE_BYTES),
    'args': (4096, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (70000, MemConstraint.ABSOLUTE_BYTES),
    # NOTE we need a deeper stack: since they make deep func calls
    'stack': (1024, MemConstraint.ABSOLUTE_BYTES),
}


def eval_interp_cpu(args, target, samples, results):
    mod = model_util.build_relay_mod(args.cifar10_conv_op_impl)
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        results['interp'] = util.eval_relay_intrp(
            mod.mod, [image_np] + [mod.params[param.name_hint] for param in mod.mod['main'].params[1:]])
        results['cpu_graph'] = util.eval_cpu_graph_runtime(mod.mod, mod.params, {'data': image_np})


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
            util.MockCMod(CIFAR10_SRC_PATH),
            cmsis_dev_config,
            lib_src_paths=CMSIS_SRC_PATHS,
            lib_headers=util.CMSIS_HEADERS,
            lib_include_paths=util.CMSIS_INCLUDE_PATHS + [CIFAR10_INCLUDE_PATH])
        micro_func = micro_mod['arm_cifar10_cnn_wrapper']
        ctx = tvm.micro_dev(0)

        output_tvm = tvm.nd.array(np.zeros(OUTPUT_SHAPE, dtype=OUT_DTYPE), ctx)

        results = []
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            label = sample['label']

            data_np = data_nt.with_layout(DATA_LAYOUT).data
            assert data_np.shape == DATA_SHAPE
            assert data_np.dtype == IN_DTYPE
            data_tvm = tvm.nd.array(data_np, ctx=ctx)

            exec_time = util.benchmark_micro_func(
                sess, micro_func, [data_tvm, output_tvm], 1, 0) #time_overhead)
            LOGGER.info(f'  model execution took {exec_time} milliseconds')

            cmsis_output_np = output_tvm.asnumpy()
            LOGGER.info(f'  output: {cmsis_output_np}')
            prediction = CIFAR10_CLASSES[np.argmax(cmsis_output_np)]
            label = CIFAR10_CLASSES[label]
            LOGGER.info(f'  prediction was {prediction}')
            LOGGER.info(f'  actual was {label}')

            results.append(cmsis_output_np)
    return results


KERNEL_LAYOUTS = {
    'conv2d_direct': 'HWIO',
    'conv2d_direct_simd': 'HWOI',
}

def eval_prebuilt_micro(args, target, samples):
    CMSIS_SEC_CONTRAINTS = {
        'text': (20000, MemConstraint.ABSOLUTE_BYTES),
        'rodata': (4096, MemConstraint.ABSOLUTE_BYTES),
        'data': (100000, MemConstraint.ABSOLUTE_BYTES),
        'bss': (644, MemConstraint.ABSOLUTE_BYTES),
        'args': (4096, MemConstraint.ABSOLUTE_BYTES),
        'heap': (100.0, MemConstraint.WEIGHT),
        'workspace': (52000, MemConstraint.ABSOLUTE_BYTES),
        # NOTE we need a deeper stack: since they make deep func calls
        'stack': (288, MemConstraint.ABSOLUTE_BYTES),
    }
    prebuilt_dev_config = generate_config(args.openocd_server_hostport, CMSIS_SEC_CONTRAINTS)
    util.reset_gdbinit(prebuilt_dev_config)

    DATA_LAYOUT = 'NHWC'
    DATA_SHAPE = (1, 32, 32, 3)
    OUTPUT_SHAPE = (10,)

    IN_DTYPE = 'uint8'
    OUT_DTYPE = 'int8'

    # Begin a session
    LOGGER.debug('[Initting]')
    mod, params = cifar10_cnn.gen_cifar10_cnn(
        data_layout, kernel_layouts,
        op_strategy=op_strategy,
        use_random_params=USE_RANDOM_PARAMS)
    # TODO probably need a different dev conf for cmsis
    with micro.Session(prebuilt_dev_config) as sess:
        # Build the function
        LOGGER.debug('[Building]')

        def build_graph_mod():
            return util.relay_micro_build(
                mod['main'],
                prebuilt_dev_config, target,
                params=params,
                lib_headers=MICRO_HEADERS,
                lib_include_paths=MICRO_INCLUDE_PATHS)

        with TARGET:
            if USE_TUNED_SCHEDULES:
                with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
                    graph_mod = build_graph_mod()
        micro_mod = create_micro_mod(
            util.MockCMod("cifar10_cnn.c"),
            prebuilt_dev_config,
            lib_src_paths=[],
            lib_headers=MICRO_HEADERS,
            lib_include_paths=util.CMSIS_INCLUDE_PATHS)

        graph_dump_json_path = (
            f'{util.get_repo_root()}/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/_tvmdbg_graph_dump.json')
        with open(graph_dump_json_path) as json_f:
            graph = json.load(json_f)
        mod = graph_runtime.create(graph, micro_mod, ctx)
        micro_func = micro_mod['arm_cifar10_cnn_wrapper']
        ctx = tvm.micro_dev(0)

        output_tvm = tvm.nd.array(np.zeros(OUTPUT_SHAPE, dtype=OUT_DTYPE), ctx)

        results = []
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            label = sample['label']

            data_np = data_nt.with_layout(DATA_LAYOUT).data
            assert data_np.shape == DATA_SHAPE
            assert data_np.dtype == IN_DTYPE
            data_tvm = tvm.nd.array(data_np, ctx=ctx)

            exec_time = util.benchmark_micro_func(
                sess, micro_func, [data_tvm, output_tvm], 1, 0) #time_overhead)
            LOGGER.info(f'  model execution took {exec_time} milliseconds')

            cmsis_output_np = output_tvm.asnumpy()
            LOGGER.info(f'  output: {cmsis_output_np}')
            prediction = CIFAR10_CLASSES[np.argmax(cmsis_output_np)]
            label = CIFAR10_CLASSES[label]
            LOGGER.info(f'  prediction was {prediction}')
            LOGGER.info(f'  actual was {label}')

            results.append(cmsis_output_np)
    return results


MICRO_SEC_CONSTRAINTS = {
    'text': (23000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (300, MemConstraint.ABSOLUTE_BYTES),
    'data': (0x80, MemConstraint.ABSOLUTE_BYTES),
    'bss': (800, MemConstraint.ABSOLUTE_BYTES),
    'args': (4096, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (145000, MemConstraint.ABSOLUTE_BYTES),
    'stack': (128, MemConstraint.ABSOLUTE_BYTES),
}


def eval_micro(args, target, samples):
    micro_dev_config = generate_config(args.openocd_server_hostport, MICRO_SEC_CONSTRAINTS)
    util.reset_gdbinit(micro_dev_config)

    mod = model_util.build_relay_mod(args.cifar10_conv_op_impl)

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
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            if mod.has_simd_strategy:
                 data_nt.resize(C=4)
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
            micro_output_np = graph_mod.get_output(0).asnumpy()
            LOGGER.info(f'  output: {micro_output_np}')
            prediction = CIFAR10_CLASSES[np.argmax(micro_output_np)]
            label = CIFAR10_CLASSES[label]
            LOGGER.info(f'  prediction was {prediction}')
            LOGGER.info(f'  actual was {label}')


def load_outputs(path):
    with open(path, 'rb') as f:
        outputs = relay.load_param_dict(f.read())
        res_outputs = {}
        for key, val in outputs.items():
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', choices=['cmsis', 'micro'], nargs='+', help='models to evaluate')
    parser.add_argument('--use-tuned-schedule', nargs='?', default='arm.stm32f746xx.e2e.log',
                        help=('Use a tuned schedule in evaluating the micro model. The path to the '
                              'tuned log can be given; if not, the default symlink generated by '
                              'tune_relay_microtvm.py is used.'))
    parser.add_argument('--openocd-server-hostport',
                        # NOTE: need to explicitly choose ipv4 address, not localhost.
                        default='127.0.0.1:6666',
                        help='Address of the OpenOCD TCL server to use for device communication.')
    parser.add_argument('--debug-runtime', action='store_true',
                        help='Use debug runtime and print graph debugging info.')
    parser.add_argument('--validate', action='store_true',
                        help='Validate on-device output against CPU graph runtime')
    model_util.define_cifar10_conv_op_impl(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    util.DEBUG_MODE = args.debug_runtime
    LOGGER.info('Run: %r', sys.argv)

    target = tvm.target.create('c -device=micro_dev')

    samples = get_sample_points(NUM_SAMPLES)

    results = {}
    for model_name in args.models:
        results[model_name] = globals()[f'eval_{model_name}'](args, target, samples)

    if args.validate:
        eval_interp_cpu(args, target, samples, results)
        print(results['interp'])
        print()
        print(results['interp'])

        print('intrp matches CPU? ' + str(np.allclose(results['interp'].astype('float32'), results['interp'].astype('float32'))))
        print('micro matches intrp? ' + str(np.allclose(results['micro'].astype('float32'), results['interp'].astype('float32'))))

    if args.debug_runtime:
        micro_outputs = load_outputs(f'{util.get_repo_root()}/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
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
        repo_debug_dir = f'{util.get_repo_root()}/debug/cpu/_tvmdbg_ctx_CPU_0'
        cpu_outputs = load_outputs(f'{repo_debug_dir}/output_tensors.params')
        save_outputs_json(cpu_outputs, 'cpu_output.json')

        micro_outputs = load_outputs(f'{repo_debug_dir}/output_tensors.params')
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
