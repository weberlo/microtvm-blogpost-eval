import json
import os
import warnings

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import numpy as np

import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.micro import create_micro_mod
from tvm.relay.testing import resnet

import micro_eval
from micro_eval.util import gen_cifar10_cnn, relay_micro_build, eval_cpu_graph_runtime, eval_relay_intrp, reset_gdbinit

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
        data = data.as_in_context(ctx).asnumpy()
        cmsis_data = data
        micro_data = np.transpose(data, (0, 3, 1, 2))
        label = int(label.asnumpy()[0])
        samples.append({'cmsis_data': cmsis_data, 'micro_data': micro_data, 'label': label})
    return samples


################
# CMSIS CONFIG #
################
if 'CMSIS_PATH' not in os.environ:
    raise RuntimeError('must have "CMSIS_PATH" in environment')
CMSIS_PATH = os.environ['CMSIS_PATH']

class DummyCMod:
    def __init__(self):
        pass

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, f'{os.path.dirname(__file__)}/../../src/cmsis_cifar10_cnn/cmsis_cifar10_cnn.c')

CMSIS_INCLUDE_PATHS = [
    f'{CMSIS_PATH}/CMSIS/Core/Include',
    f'{CMSIS_PATH}/CMSIS/DSP/Include',
    f'{CMSIS_PATH}/CMSIS/NN/Include',
    f'{os.path.dirname(__file__)}/../../src/cmsis_cifar10_cnn'
]

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


###############
# µTVM CONFIG #
###############
from collections import OrderedDict
from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint
TVM_DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6668)
TVM_DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
    #('text', (19000, MemConstraint.ABSOLUTE_BYTES)),
    ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (100.0, MemConstraint.WEIGHT)),
    ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    ]))
CMSIS_DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6668)
CMSIS_DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
    #('text', (10000, MemConstraint.ABSOLUTE_BYTES)),
    ('text', (16000, MemConstraint.ABSOLUTE_BYTES)),
    ('rodata', (4096, MemConstraint.ABSOLUTE_BYTES)),
    ('data', (128000, MemConstraint.ABSOLUTE_BYTES)),
    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (100.0, MemConstraint.WEIGHT)),
    ('workspace', (47360, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    ]))
TARGET = tvm.target.create('c -device=micro_dev')


#############
# Debugging #
#############
reset_gdbinit(TVM_DEV_CONFIG)


###############
# Main Course #
###############
CIFAR10_CLASSES = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
NUM_SAMPLES = 5

USE_TUNED_SCHEDULES = True
USE_RANDOM_PARAMS = False

TVM_DATA_LAYOUT = 'NCHW'
CMSIS_DATA_LAYOUT = 'NHWC'
CMSIS_DATA_SHAPE = (1, 32, 32, 3)
CMSIS_OUTPUT_SHAPE = (10,)
IN_DTYPE = 'uint8'
OUT_DTYPE = 'int8'

assert False, "execute an empty kernel to figure out what the comm overhead is and subtract that from the times"

def eval_cmsis(samples):
    # Begin a session
    print("[Initting]")
    # TODO probably need a different dev conf for cmsis
    with micro.Session(CMSIS_DEV_CONFIG) as sess:
        # Build the function
        print("[Building]")
        micro_mod = create_micro_mod(
            DummyCMod(),
            CMSIS_DEV_CONFIG,
            lib_src_paths=CMSIS_SRC_PATHS,
            lib_include_paths=CMSIS_INCLUDE_PATHS)
        micro_func = micro_mod['arm_cifar10_cnn_wrapper']
        ctx = tvm.micro_dev(0)

        output_tvm = tvm.nd.array(np.zeros(CMSIS_OUTPUT_SHAPE, dtype=OUT_DTYPE), ctx)

        with open('cmsis_results.txt', 'w') as f:
            for i, sample in enumerate(samples):
                f.write('[Sample {i}]\n')
                image_np = sample['cmsis_data']
                label = sample['label']

                assert image_np.shape == CMSIS_DATA_SHAPE
                assert image_np.dtype == IN_DTYPE
                data_tvm = tvm.nd.array(image_np, ctx=ctx)

                # sync before and after to ensure these are the only tasks in the queue
                ctx.sync()
                sess.get_last_batch_time()
                sess.get_last_batch_cycles()
                micro_func(data_tvm, output_tvm)
                ctx.sync()
                exec_time = sess.get_last_batch_time()
                exec_cycles = sess.get_last_batch_cycles()
                f.write(f'  Model execution took {exec_time} milliseconds and {exec_cycles} cycles\n')

                cmsis_output_np = output_tvm.asnumpy()
                f.write(f'  Output: {cmsis_output_np}\n')
                prediction = CIFAR10_CLASSES[np.argmax(cmsis_output_np)]
                label = CIFAR10_CLASSES[label]
                f.write(f'  Prediction was {prediction}\n')
                f.write(f'  Actual was {label}\n')


def eval_micro(samples):
    mod, params = gen_cifar10_cnn(USE_RANDOM_PARAMS)
    # Begin a session
    print("[Initting]")
    with micro.Session(TVM_DEV_CONFIG) as sess:
        # Build the function
        print("[Building]")
        from tvm import autotvm
        DEVICE_ID = 'arm.stm32f746xx'
        E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'
        if USE_TUNED_SCHEDULES:
            #assert False, "will apply history best use the best in the log file? can we have multiple entries for the same workload?"
            with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
                with TARGET:
                    graph_mod = relay_micro_build(mod['main'], TVM_DEV_CONFIG, TARGET, params=params)
            log_file_name = 'micro_tuned_results.txt'
        else:
            graph_mod = relay_micro_build(mod['main'], TVM_DEV_CONFIG, TARGET, params=params)
            log_file_name = 'micro_untuned_results.txt'

        with open(log_file_name, 'w') as f:
            for i, sample in enumerate(samples):
                f.write(f'[Sample {i}]\n')
                #from PIL import Image
                #img = Image.fromarray(np.transpose(image_np, (0, 2, 3, 1)).reshape((image_np.shape[2], image_np.shape[3], image_np.shape[1])), 'RGB')
                #img.save('sample.png')
                image_np = sample['micro_data']
                label = sample['label']

                assert image_np.shape == tuple(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))

                # Execute with `image` as the input.
                print("[Executing]")
                sess.get_last_batch_time()
                sess.get_last_batch_cycles()
                ctx = tvm.micro_dev(0)
                ctx.sync()
                graph_mod.run(data=image_np)
                ctx.sync()
                exec_time = sess.get_last_batch_time()
                exec_cycles = sess.get_last_batch_cycles()
                f.write(f'  Model execution took {exec_time} milliseconds and {exec_cycles} cycles\n')

                # Get output
                micro_output_np = graph_mod.get_output(0).asnumpy()
                f.write(f'  Output: {micro_output_np}\n')
                prediction = CIFAR10_CLASSES[np.argmax(micro_output_np)]
                label = CIFAR10_CLASSES[label]
                f.write(f'  Prediction was {prediction}\n')
                f.write(f'  Actual was {label}\n')
                #print('BREAKING EARLY')
                #break


#samples = get_sample_points(NUM_SAMPLES)
#eval_cmsis(samples)
#eval_micro(samples)

#with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
#    intrp_output_np = eval_relay_intrp(
#        mod, [image_np] + [params[param.name_hint] for param in mod['main'].params[1:]])
#    cpu_graph_output_np = eval_cpu_graph_runtime(mod, params, {'data': image_np})
#print(intrp_output_np)
#print()
#print(cpu_graph_output_np)

#print('intrp matches CPU? ' + str(np.allclose(intrp_output_np.astype('float32'), cpu_graph_output_np.astype('float32'))))
#print('micro matches intrp? ' + str(np.allclose(micro_output_np.astype('float32'), intrp_output_np.astype('float32'))))

with open('cmsis_results.txt', 'r') as f:
    print('[[CMSIS]]')
    print(f.read())

if USE_TUNED_SCHEDULES:
    with open('micro_tuned_results.txt', 'r') as f:
        print('[[Micro Tuned]]')
        print(f.read())
else:
    with open('micro_untuned_results.txt', 'r') as f:
        print('[[Micro Untuned]]')
        print(f.read())

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
