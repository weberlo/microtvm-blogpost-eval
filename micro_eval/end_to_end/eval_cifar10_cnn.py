import json
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
from tvm.relay.testing import resnet

import micro_eval
from micro_eval.util import relay_micro_build, eval_cpu_graph_runtime, eval_relay_intrp

LAYOUT = 'NCHW'

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transforms.ToTensor()),
    batch_size=64, shuffle=True)

def get_sample_points(n):
    """Grabs a single input/label pair from MNIST"""
    ctx = mx.cpu()
    # Load a random image from the test dataset
    sample_data = mx.gluon.data.DataLoader(
            #mx.gluon.data.vision.CIFAR10(train=False, transform=transform),
            mx.gluon.data.vision.CIFAR10(train=False),
            1,
            shuffle=True)
    samples = []
    for i, (data, label) in zip(range(n), sample_data):
        if i == n:
            break
        data = nd.transpose(data, (0, 3, 1, 2))
        data = data.as_in_context(ctx).asnumpy()
        label = int(label.asnumpy()[0])
        samples.append((data, label))
    return samples


##########
# CONFIG #
##########
from collections import OrderedDict
from tvm.micro.device.arm import stm32f746xx
from tvm.micro.device.arm.stm32f746xx import MemConstraint
DEV_CONFIG = stm32f746xx.default_config('127.0.0.1', 6669)
DEV_CONFIG['mem_layout'] = stm32f746xx.gen_mem_layout(OrderedDict([
    ('text', (10000, MemConstraint.ABSOLUTE_BYTES)),
    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
    ('heap', (100.0, MemConstraint.WEIGHT)),
    ('workspace', (132000, MemConstraint.ABSOLUTE_BYTES)),
    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    ]))
TARGET = tvm.target.create('c -device=micro_dev')


#############
# Debugging #
#############
def reset_gdbinit():
    if 'server_port' not in DEV_CONFIG:
        return
    with open('/home/lweber/gdb-conf/.gdbinit', 'w') as f:
        gdb_port = DEV_CONFIG['server_port'] - 3333
        gdbinit_contents = (
f"""layout src
target remote localhost:{gdb_port}
set $pc = UTVMInit
break UTVMDone

define print_utvm_args
    set $i = 0
    while $i < utvm_num_tasks
        set $j = 0
        eval "print \\"TASK %d ARGS\\"", $i
        eval "set $num_task_args = utvm_tasks[$i].num_args"
        print "num_args: %d", $num_task_args
        while $j < $num_task_args
            eval "set $num_bits = ((TVMArray*) utvm_tasks[0].arg_values[0].v_handle)->dtype.bits"
            if $num_bits == 8
                print "dtype: int8"
                eval "p/d *((int8_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            if $num_bits == 32
                print "dtype: int32"
                eval "p/d *((int32_t*) ((TVMArray*) utvm_tasks[$i].arg_values[$j].v_handle)->data)@16"
            end
            set $j = $j + 1
        end
        set $i = $i + 1
    end
end

print_utvm_args
""")
        f.write(gdbinit_contents)
reset_gdbinit()

# TODO adding a `clip` op causes free variables to be detected in the module
mod = relay.fromtext("""
v0.0.4
def @main(%data: Tensor[(1, 3, 32, 32), uint8],
    %mean_data: Tensor[(1, 3, 32, 32), uint8],
    %conv0_weight: Tensor[(32, 3, 5, 5), int8],
    %conv0_bias: Tensor[(32), int8],
    %conv1_weight: Tensor[(32, 32, 5, 5), int8],
    %conv1_bias: Tensor[(32), int8],
    %conv2_weight: Tensor[(64, 32, 5, 5), int8],
    %conv2_bias: Tensor[(64), int8],
    %dense0_weight: Tensor[(10, 1024), int8],
    %dense0_bias: Tensor[(10), int8]) {
  %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
  %1 = nn.conv2d(%0, %conv0_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
  %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"));
  %3 = right_shift(%2, 9);
  %4 = cast(%3, "int8");
  %5 = nn.max_pool2d(%4, pool_size=[3, 3], strides=[2, 2], ceil_mode=True);
  %6 = nn.relu(%5);
  %7 = nn.conv2d(%6, %conv1_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
  %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"));
  %9 = right_shift(%8, 9);
  %10 = cast(%9, "int8");
  %11 = nn.relu(%10);
  %12 = nn.avg_pool2d(%11, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
  %13 = nn.conv2d(%12, %conv2_weight, padding=[2, 2], channels=64, kernel_size=[5, 5], out_dtype="int32");
  %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"));
  %15 = right_shift(%14, 9);
  %16 = cast(%15, "int8");
  %17 = nn.relu(%16);
  %18 = nn.avg_pool2d(%17, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
  %19 = nn.batch_flatten(%18);
  %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
  %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
  %22 = right_shift(%21, 5);
  cast(%22, "int8")
}
""")

USE_RANDOM_PARAMS = False

## generate random input
#param = mod['main'].params[0]
#shape = list(map(lambda x: x.value, param.checked_type.shape))
#dtype = param.checked_type.dtype
#assert 'data' in param.name_hint
#image_np = tvm.nd.array(np.random.randint(80, 180, size=shape, dtype=dtype), tvm.cpu(0))
samples = get_sample_points(15)

if USE_RANDOM_PARAMS:
    # generate random params
    params = {}
    for param in mod['main'].params[1:]:
        shape = list(map(lambda x: x.value, param.checked_type.shape))
        dtype = param.checked_type.dtype
        if 'bias' in param.name_hint:
            result = tvm.nd.array(np.random.randint(-3, 3, size=shape, dtype=dtype), tvm.cpu(0))
        elif 'weight' in param.name_hint:
            result = tvm.nd.array(np.random.randint(-30, 30, size=shape, dtype=dtype), tvm.cpu(0))
        elif 'mean' in param.name_hint:
            result = tvm.nd.array(np.random.randint(130, 140, size=shape, dtype=dtype), tvm.cpu(0))
        else:
            assert False
        params[param.name_hint] = result
else:
    with open('cifar10_cnn_params.json', 'r') as f:
        params = json.load(f)
    for formal_param in mod['main'].params[1:]:
        shape = list(map(lambda x: x.value, formal_param.checked_type.shape))
        dtype = formal_param.checked_type.dtype
        name = formal_param.name_hint
        # NCHW -> NHWC
        orig_np = np.array(params[name]).astype(dtype)
        print(name)
        print(orig_np.shape)
        if name == 'mean_data':
            # NCHW
            # N == 0
            # C == 1
            # H == 2
            # W == 3
            # NHWC (0, 2, 3, 1)

            # NHWC
            # N == 0
            # H == 1
            # W == 2
            # C == 3
            # NCHW (0, 3, 1, 2)
            orig_np = orig_np.reshape((shape[0], shape[2], shape[3], shape[1]))
            print(orig_np.shape)
            orig_np = orig_np.transpose(0, 3, 1, 2)
            print(orig_np.shape)
        elif name == 'conv0_weight':
            # CO, CI, KW, KH
            # CO == 0
            # CI == 1
            # KW == 2
            # KH == 3
            # CI, KW, KH, CO (1, 2, 3, 0)

            # CI, KW, KH, CO
            # CI == 0
            # KW == 1
            # KH == 2
            # CO == 3
            # CO, CI, KW, KH (3, 0, 1, 2)
            orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
            print(orig_np.shape)
            orig_np = orig_np.transpose(3, 0, 1, 2)
            print(orig_np.shape)
        elif name == 'conv0_bias':
            pass
        elif name == 'conv1_weight':
            orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
            print(orig_np.shape)
            orig_np = orig_np.transpose(3, 0, 1, 2)
            print(orig_np.shape)
        elif name == 'conv1_bias':
            pass
        elif name == 'conv2_weight':
            orig_np = orig_np.reshape((shape[1], shape[2], shape[3], shape[0]))
            print(orig_np.shape)
            orig_np = orig_np.transpose(3, 0, 1, 2)
            print(orig_np.shape)
        elif name == 'conv2_bias':
            pass
        elif name == 'dense0_weight':
            orig_np = orig_np.reshape((shape[1], shape[0]))
            print(orig_np.shape)
            orig_np = orig_np.transpose(1, 0)
            print(orig_np.shape)
        elif name == 'dense0_bias':
            pass
        else:
            assert False
        #assert False, "TODO do layout transformations according to multiplication ordering in the CIFAR10 cpp file"
        params[name] = tvm.nd.array(orig_np, tvm.cpu(0))


# Begin a session
print("[Initting]")
with micro.Session(DEV_CONFIG) as sess:
    # Build the function
    print("[Building]")
    from tvm import autotvm
    DEVICE_ID = 'arm.stm32f746xx'
    E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'
    #with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
    #    with TARGET:
    #        graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, TARGET, params=params)
    graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, TARGET, params=params)

    for image_np, label in samples:
        #from PIL import Image
        #img = Image.fromarray(np.transpose(image_np, (0, 2, 3, 1)).reshape((image_np.shape[2], image_np.shape[3], image_np.shape[1])), 'RGB')
        #img.save('sample.png')

        # Execute with `image` as the input.
        print("[Executing]")
        sess.get_last_batch_time()
        ctx = tvm.micro_dev(0)
        ctx.sync()
        graph_mod.run(data=image_np)
        ctx.sync()
        exec_time = sess.get_last_batch_time()
        print(f'  Model execution took {exec_time} milliseconds')

        # Get output
        CIFAR10_CLASSES = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        micro_output_np = graph_mod.get_output(0).asnumpy()
        print(f'output: {micro_output_np}')
        prediction = CIFAR10_CLASSES[np.argmax(micro_output_np)]
        label = CIFAR10_CLASSES[label]
        print(f'  Prediction was {prediction}')
        input(f'  Actual was {label} ')
        #print('BREAKING EARLY')
        #break


#with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
#    intrp_output_np = eval_relay_intrp(
#        mod, [image_np] + [params[param.name_hint] for param in mod['main'].params[1:]])
#    cpu_graph_output_np = eval_cpu_graph_runtime(mod, params, {'data': image_np})
#print(intrp_output_np)
#print()
#print(cpu_graph_output_np)

#print('intrp matches CPU? ' + str(np.allclose(intrp_output_np.astype('float32'), cpu_graph_output_np.astype('float32'))))
#print('micro matches intrp? ' + str(np.allclose(micro_output_np.astype('float32'), intrp_output_np.astype('float32'))))


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

#print('#######################')
#print('#######################')
#print('##### CPU OUTPUTS #####')
#print('#######################')
#print('#######################')
#print(sorted(cpu_outputs.keys()))
#
print('########################')
print('########################')
print('##### MICRO PARAMS #####')
print('########################')
print('########################')
print(sorted(micro_outputs.keys()))
