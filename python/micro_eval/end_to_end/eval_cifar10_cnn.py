import json
import os
import warnings
from collections import OrderedDict

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
import tvm.micro as micro
from tvm import autotvm
from tvm.micro import create_micro_mod
from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint
from tvm.relay.testing import resnet
from topi.util import get_const_tuple

import micro_eval
from micro_eval.util import (
    CMSIS_PATH, CMSIS_INCLUDE_PATHS,
    MockCMod,
    get_logger,
    relay_micro_build,
    benchmark_micro_func,
    eval_cpu_graph_runtime, eval_relay_intrp,
    NamedTensor, NamedType, BakedType,
    deep_tuple,
    reset_gdbinit, get_comm_overhead
)
from micro_eval.micro_topi import ManualConfigContext, ManualConfigSpace, collect_conv_workloads
from micro_eval.micro_topi.cortex_m7.conv2d.direct import (
    conv2d_direct, conv2d_direct_compute, conv2d_direct_nhwc_schedule
)
from micro_eval.micro_topi.cortex_m7.conv2d.direct_simd import (
    conv2d_direct_simd, conv2d_direct_simd_compute, conv2d_direct_simd_nhwc_schedule
)
# import micro_eval.micro_topi.cortex_m7.conv2d.partial_im2col
from micro_eval.model.cifar10_cnn import gen_cifar10_cnn

LOGGER = get_logger('cifar10_cnn.log')

TARGET = tvm.target.create('c -device=micro_dev')
DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6666)

###################
# MODEL/DATA UTIL #
###################
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
        data_nt = NamedTensor(data_np, 'NHWC')
        label = int(label.asnumpy()[0])
        samples.append({'data': data_nt, 'label': label})
    return samples

################
# CMSIS CONFIG #
################
CIFAR10_SRC_PATH = f'{os.path.dirname(__file__)}/../../../cmsis_src/cmsis_cifar10_cnn/cmsis_cifar10_cnn.c'
CIFAR10_INCLUDE_PATH = f'{os.path.dirname(__file__)}/../../../cmsis_src/cmsis_cifar10_cnn'
CMSIS_SRC_PATHS = [
    f'{CMSIS_PATH}/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c',
    f'{CMSIS_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c',
]

##############
# EVALUATION #
##############
CIFAR10_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
NUM_SAMPLES = 10

def eval_cmsis(samples, time_overhead):
    DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
        ('text', (16000, MemConstraint.ABSOLUTE_BYTES)),
        ('rodata', (4096, MemConstraint.ABSOLUTE_BYTES)),
        ('data', (100000, MemConstraint.ABSOLUTE_BYTES)),
        ('bss', (644, MemConstraint.ABSOLUTE_BYTES)),
        ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
        ('heap', (100.0, MemConstraint.WEIGHT)),
        ('workspace', (52000, MemConstraint.ABSOLUTE_BYTES)),
        # NOTE we need a deeper stack, since they make deep func calls
        ('stack', (288, MemConstraint.ABSOLUTE_BYTES)),
        ]))
    reset_gdbinit(DEV_CONFIG)

    DATA_LAYOUT = 'NHWC'
    DATA_SHAPE = (1, 32, 32, 3)
    OUTPUT_SHAPE = (10,)

    IN_DTYPE = 'uint8'
    OUT_DTYPE = 'int8'

    # Begin a session
    print('[Initting]')
    # TODO probably need a different dev conf for cmsis
    with micro.Session(DEV_CONFIG) as sess:
        # Build the function
        print('[Building]')
        micro_mod = create_micro_mod(
            MockCMod(CIFAR10_SRC_PATH),
            DEV_CONFIG,
            lib_src_paths=CMSIS_SRC_PATHS,
            lib_include_paths=CMSIS_INCLUDE_PATHS + [CIFAR10_INCLUDE_PATH])
        micro_func = micro_mod['arm_cifar10_cnn_wrapper']
        ctx = tvm.micro_dev(0)

        output_tvm = tvm.nd.array(np.zeros(OUTPUT_SHAPE, dtype=OUT_DTYPE), ctx)

        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            label = sample['label']

            data_np = data_nt.with_layout(DATA_LAYOUT).data
            assert data_np.shape == DATA_SHAPE
            assert data_np.dtype == IN_DTYPE
            data_tvm = tvm.nd.array(data_np, ctx=ctx)

            exec_time = benchmark_micro_func(
                sess, micro_func, [data_tvm, output_tvm], 1, time_overhead)
            LOGGER.info(f'  model execution took {exec_time} milliseconds')

            cmsis_output_np = output_tvm.asnumpy()
            LOGGER.info(f'  output: {cmsis_output_np}')
            prediction = CIFAR10_CLASSES[np.argmax(cmsis_output_np)]
            label = CIFAR10_CLASSES[label]
            LOGGER.info(f'  prediction was {prediction}')
            LOGGER.info(f'  actual was {label}')


def gen_model_config(mod):
    convs = collect_conv_workloads(mod['main'])

    def gen_direct_simd_cfg(M, K, N):
        from tvm.autotvm.task.space import ReorderEntity, SplitEntity, OtherOptionEntity
        cfg = ManualConfigSpace()
        cfg.template_key = 'direct_simd'
        cfg['tile_ow'] = SplitEntity([-1, M])
        cfg['tile_ci'] = SplitEntity([-1, K])
        cfg['tile_co'] = SplitEntity([-1, N])
        # TODO we shouldn't need to mirror the order of the axes
        # specified in the config space definition to mock a reordering
        # here
        reorder_base = ['n', 'oh', 'owo', 'owi', 'coo', 'coi', 'kh', 'kw', 'cio', 'cii']
        reorder_target = ['n', 'oh', 'kh', 'kw', 'owo', 'coo', 'cio', 'owi', 'coi', 'cii']
        cfg['reorder_0_simd'] = ReorderEntity(
            [reorder_base.index(axis) for axis in reorder_target])
        cfg['auto_unroll_max_step'] = OtherOptionEntity(0)
        cfg['unroll_explicit'] = OtherOptionEntity(0)
        return cfg

    assert len(convs) == 3
    result = {}
    result[(TARGET, convs[0])] = FallbackConfigEntity()
    result[(TARGET, convs[1])] = gen_direct_simd_cfg(8, 32, 8)
    result[(TARGET, convs[2])] = gen_direct_simd_cfg(8, 32, 8)
    return result


def eval_micro(samples, time_overhead):
    USE_TUNED_SCHEDULES = True
    USE_RANDOM_PARAMS = True

    DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
        #('text', (19000, MemConstraint.ABSOLUTE_BYTES)),
        ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
        ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
        ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
        ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
        ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
        ('heap', (100.0, MemConstraint.WEIGHT)),
        ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
        ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
        ]))
    reset_gdbinit(DEV_CONFIG)

    # per-conv op strategies (first entry is the strategy of the first conv and so on).
    # we want the ability to configure the op strategy, instead of just using
    # the best strategy in the log, because certain strategy combos have a
    # memory footprint that exceeds the available memory of the device.
    OP_STRATEGIES = [
        # conv2d_partial_im2col,
        # conv2d_direct_simd,
        # conv2d_direct_simd,
        conv2d_direct,
        conv2d_direct_simd,
        conv2d_direct_simd,
        ]
    data_layout = OP_STRATEGIES[0].default_data_layout
    kernel_layouts = []
    for strat in OP_STRATEGIES:
        assert strat.default_data_layout == data_layout, 'data layouts for all convs must agree'
        kernel_layouts.append(strat.default_kernel_layout)

    # USE_SIMD = True
    # if USE_SIMD:
    #     # don't use SIMD layout in the first layer
    #     KERNEL_LAYOUTS = ['HWIO', 'HWOI', 'HWOI']
    # else:
    #     DATA_LAYOUT = 'NHWC'
    #     KERNEL_LAYOUTS = 'HWIO'

    DEVICE_ID = 'arm.stm32f746xx'
    E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'
    # E2E_LOG_FILE_NAME = f'autotvm_logs/pre_simd/{DEVICE_ID}.e2e.log.manually_fixed'

    # op_strategy = 'direct_simd'
    op_strategy = 'direct'
    # op_strategy = 'direct_simd' if USE_SIMD else 'direct'
    mod, params = gen_cifar10_cnn(
        data_layout, kernel_layouts,
        op_strategy=op_strategy,
        use_random_params=USE_RANDOM_PARAMS)
    if not USE_TUNED_SCHEDULES:
        model_config = gen_model_config(mod)
    print('[Initting]')
    with micro.Session(DEV_CONFIG) as sess:
        print('[Building]')
        def build_graph_mod():
            return relay_micro_build(
                mod['main'],
                DEV_CONFIG, TARGET,
                params=params,
                lib_include_paths=CMSIS_INCLUDE_PATHS)

        with TARGET:
            if USE_TUNED_SCHEDULES:
                with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
                    graph_mod = build_graph_mod()
            else:
                with ManualConfigContext(model_config):
                    graph_mod = build_graph_mod()

        if USE_TUNED_SCHEDULES:
            LOGGER.info('[[Micro Tuned]]')
        else:
            LOGGER.info('[[Micro Untuned]]')
        for i, sample in enumerate(samples):
            LOGGER.info(f'[Sample {i}]')
            data_nt = sample['data']
            # if OP_STRATEGIES[0] in (conv2d_direct_simd, conv2d_partial_im2col):
            #     data_nt.resize(C=4)
            data_np = data_nt.with_layout(data_layout).data
            label = sample['label']

            assert data_np.shape == get_const_tuple(mod['main'].params[0].checked_type.shape)

            # execute with `image` as the input
            print('[Executing]')
            sess.get_last_batch_time()
            ctx = tvm.micro_dev(0)
            ctx.sync()
            graph_mod.run(data=data_np)
            ctx.sync()
            exec_time = sess.get_last_batch_time() - time_overhead
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


def main():
    time_overhead = get_comm_overhead(DEV_CONFIG, num_trials=5)
    LOGGER.info(f'time overhead is {time_overhead}ms')
    # time_overhead = 0.0

    samples = get_sample_points(NUM_SAMPLES)
    eval_cmsis(samples, time_overhead)
    eval_micro(samples, time_overhead)

    #with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    #    intrp_output_np = eval_relay_intrp(
    #        mod, [image_np] + [params[param.name_hint] for param in mod['main'].params[1:]])
    #    cpu_graph_output_np = eval_cpu_graph_runtime(mod, params, {'data': image_np})
    #print(intrp_output_np)
    #print()
    #print(cpu_graph_output_np)

    #print('intrp matches CPU? ' + str(np.allclose(intrp_output_np.astype('float32'), cpu_graph_output_np.astype('float32'))))
    #print('micro matches intrp? ' + str(np.allclose(micro_output_np.astype('float32'), intrp_output_np.astype('float32'))))

    with open('cifar10_cnn.log', 'r') as f:
        print(f.read())

    if micro_eval.util.DEBUG_MODE:
        micro_outputs = load_outputs('/home/lweber/microtvm-blogpost-eval/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
        save_outputs_json(micro_outputs, 'micro_output.json')

        print(f'micro output keys: {micro_outputs.keys()}')
        for key in micro_outputs.keys():
            micro_val = micro_outputs[key]
            print()
            print()
            print(f'=================[{key}]===================')
            print(micro_val)
            input('========================================')

    #if micro_eval.util.DEBUG_MODE:
    #    cpu_outputs = load_outputs('/home/lweber/microtvm-blogpost-eval/debug/cpu/_tvmdbg_ctx_CPU_0/output_tensors.params')
    #    save_outputs_json(cpu_outputs, 'cpu_output.json')
    #
    #    micro_outputs = load_outputs('/home/lweber/microtvm-blogpost-eval/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
    #    save_outputs_json(micro_outputs, 'micro_output.json')
    #
    #    for key in cpu_outputs.keys():
    #        if key not in micro_outputs:
    #            print(f'{key} not found in micro output')
    #            print(f'cpu keys: {cpu_outputs.keys()}')
    #            print(f'micro keys: {micro_outputs.keys()}')
    #        cpu_val = cpu_outputs[key]
    #        micro_val = micro_outputs[key]
    #        if np.allclose(cpu_val.astype('float32'), micro_val.astype('float32')):
    #            input(f'{key} output matches!')
    #        else:
    #            print('========================================')
    #            print(cpu_val)
    #            print('----------------------------------------')
    #            print(micro_val)
    #            print('----------------------------------------')
    #            input(f'{key} output does not match!')
    #            print('========================================')
    #            print()
    #            print()


if __name__ == "__main__":
    main()
