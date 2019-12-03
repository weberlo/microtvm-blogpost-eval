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

MODEL_NAME = "cifar10_cnn"
# Set to true if you want to train before inference
SHOULD_TRAIN = False
# Batch number after which you just don't care
STOP_TRAINING_AFTER = 1

# Use GPU if one exists, else use CPU
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

batch_size = 64

LAYOUT = 'NCHW'

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

def build_cifar10_cnn(net):
    """Build a simple convolutional network that replicates ARM's CMSIS-NN example"""
    with net.name_scope():
        # First convolution
        net.add(gluon.nn.Conv2D(
            in_channels=3,
            channels=32,
            kernel_size=5,
            strides=1,
            padding=2,
            use_bias=True,
            layout=LAYOUT,
            ))
        net.add(gluon.nn.MaxPool2D(
            pool_size=3,
            strides=2,
            padding=0,
            layout=LAYOUT
            ))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.Conv2D(
            in_channels=32,
            channels=32,
            kernel_size=5,
            strides=1,
            padding=2,
            use_bias=True,
            layout=LAYOUT,
            ))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.AvgPool2D(
            pool_size=3,
            strides=2,
            padding=0,
            layout=LAYOUT
            ))
        net.add(gluon.nn.Conv2D(
            in_channels=32,
            channels=64,
            kernel_size=5,
            strides=1,
            padding=2,
            use_bias=True,
            layout=LAYOUT,
            ))
        net.add(gluon.nn.Activation('relu'))
        net.add(gluon.nn.AvgPool2D(
            pool_size=3,
            strides=2,
            padding=0,
            layout=LAYOUT
            ))
        net.add(gluon.nn.Dense(
            10,
            use_bias=True,
            ))
        return net


def train_model(model):
    """Train a given model using MNIST data"""
    # Initialize the parameters with Xavier initializer
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    # Use cross entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # Use Adam optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .001})

    # Train for one epoch
    for epoch in range(1):
        # Iterate through the images and labels in the training data
        for batch_num, (data, label) in enumerate(train_data):
            # get the images and labels
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Ask autograd to record the forward pass
            with autograd.record():
                # Run the forward pass
                output = model(data)
                # Compute the loss
                loss = softmax_cross_entropy(output, label)
            # Compute gradients
            loss.backward()
            # Update parameters
            trainer.step(data.shape[0])

            if STOP_TRAINING_AFTER > 0 and batch_num > STOP_TRAINING_AFTER:
                break

            # Print loss once in a while
            if batch_num % 50 == 0:
                curr_loss = nd.mean(loss).asscalar()
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, batch_num, curr_loss))


def get_sample_point():
    """Grabs a single input/label pair from MNIST"""
    assert False, "doesn't work with our new setup"

    def transform(data, label):
        return data.astype(np.float32)/255.0, label.astype(np.float32)

    # Load a random image from the test dataset
    sample_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=False, transform=transform),
            1,
            shuffle=True)
    for data, label in sample_data:
        #img = nd.transpose(data, (1,0,2,3))
        #img = nd.reshape(img, (28,28,1))
        #imtiles = nd.tile(img, (1,1,3))

        data = nd.transpose(data, (0, 3, 1, 2))
        data = data.as_in_context(ctx).asnumpy()
        label = int(label.asnumpy()[0])
        return data, label


if SHOULD_TRAIN:
    cifar10_cnn = build_cifar10_cnn(gluon.nn.HybridSequential())
    cifar10_cnn.hybridize()
    train_model(cifar10_cnn)
    cifar10_cnn.export(MODEL_NAME, epoch=1)

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

## Import model
#print("[Import Model]")
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    cifar10_cnn = gluon.nn.SymbolBlock.imports(f"{MODEL_NAME}-symbol.json", ['data'], f"{MODEL_NAME}-0001.params", ctx=ctx)
#
## Convert to Relay
#mod, params = relay.frontend.from_mxnet(
#    cifar10_cnn, shape={"data": (1, 3, 32, 32)})
#
#print("[Quantizing]")
#with relay.quantize.qconfig(skip_k_conv=0, round_for_shift=True):
#    mod = relay.module.Module.from_expr(relay.quantize.quantize(mod['main'], params))

mod = relay.fromtext("""
v0.0.4
def @main(%data: Tensor[(1, 3, 32, 32), uint8],
    %mean_data: Tensor[(1, 3, 32, 32), uint8],
    %conv0_weight: Tensor[(32, 3, 5, 5), int8],
    %conv0_bias: Tensor[(32), int32],
    %conv1_weight: Tensor[(32, 32, 5, 5), int8],
    %conv1_bias: Tensor[(32), int32],
    %conv2_weight: Tensor[(64, 32, 5, 5), int8],
    %conv2_bias: Tensor[(64), int32],
    %dense0_weight: Tensor[(10, 1024), int8],
    %dense0_bias: Tensor[(10), int32]) {
  %0 = nn.conv2d(%data - %mean_data, %conv0_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
  %1 = nn.bias_add(%0, %conv0_bias);
  %2 = right_shift(%1, 9);
  %3 = cast(%2, "int8");
  %4 = nn.max_pool2d(%3, pool_size=[3, 3], strides=[2, 2], ceil_mode=True);
  %5 = nn.relu(%4);
  %6 = nn.conv2d(%5, %conv1_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int32");
  %7 = nn.bias_add(%6, %conv1_bias);
  %8 = right_shift(%7, 9);
  %9 = cast(%8, "int8");
  %10 = nn.relu(%9);
  %11 = nn.avg_pool2d(%10, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
  %12 = nn.conv2d(%11, %conv2_weight, padding=[2, 2], channels=64, kernel_size=[5, 5], out_dtype="int32");
  %13 = nn.bias_add(%12, %conv2_bias);
  %14 = right_shift(%13, 9);
  %15 = cast(%14, "int8");
  %16 = nn.relu(%15);
  %17 = nn.avg_pool2d(%16, pool_size=[3, 3], strides=[2, 2], count_include_pad=True, ceil_mode=True);
  %18 = nn.batch_flatten(%17);
  %19 = nn.dense(%18, %dense0_weight, units=10, out_dtype="int32");
  nn.bias_add(%19, %dense0_bias, axis=-1)
}
""")

USE_RANDOM_PARAMS = False

# generate random input
param = mod['main'].params[0]
shape = list(map(lambda x: x.value, param.checked_type.shape))
dtype = param.checked_type.dtype
assert 'data' in param.name_hint
image_np = tvm.nd.array(np.random.randint(80, 180, size=shape, dtype=dtype), tvm.cpu(0))

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
        print(name)
        # NCHW -> NHWC
        params[name] = tvm.nd.array(np.array(params[name]).astype(dtype).reshape(shape), tvm.cpu(0))


# Begin a session
print("[Initting]")
with micro.Session(DEV_CONFIG) as sess:
    # Build the function
    print("[Building]")
    from tvm import autotvm
    DEVICE_ID = 'arm.stm32f746xx'
    E2E_LOG_FILE_NAME = f'{DEVICE_ID}.e2e.log'
    with autotvm.apply_history_best(E2E_LOG_FILE_NAME):
        with TARGET:
            graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, TARGET, params=params)

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
    micro_output_np = graph_mod.get_output(0).asnumpy()

with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
    intrp_output_np = eval_relay_intrp(
        mod, [image_np] + [params[param.name_hint] for param in mod['main'].params[1:]])
    cpu_graph_output_np = eval_cpu_graph_runtime(mod, params, {'data': image_np})
#print(intrp_output_np)
#print()
#print(cpu_graph_output_np)

#tvm.testing.assert_allclose(intrp_output_np.astype('float32'), cpu_graph_output_np.astype('float32'))
#tvm.testing.assert_allclose(micro_output_np.astype('float32'), intrp_output_np.astype('float32'))
print('intrp matches CPU? ' + str(np.allclose(intrp_output_np.astype('float32'), cpu_graph_output_np.astype('float32'))))
print('micro matches intrp? ' + str(np.allclose(micro_output_np.astype('float32'), intrp_output_np.astype('float32'))))


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
    cpu_outputs = load_outputs('/home/lweber/microtvm-blogpost-eval/debug/cpu/_tvmdbg_ctx_CPU_0/output_tensors.params')
    save_outputs_json(cpu_outputs, 'cpu_output.json')
        
    micro_outputs = load_outputs('/home/lweber/microtvm-blogpost-eval/debug/micro/_tvmdbg_ctx_MICRO_DEV_0/output_tensors.params')
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

#print('#######################')
#print('#######################')
#print('##### CPU OUTPUTS #####')
#print('#######################')
#print('#######################')
#print(sorted(cpu_outputs.keys()))
#
#print('########################')
#print('########################')
#print('##### MICRO PARAMS #####')
#print('########################')
#print('########################')
#print(sorted(micro_outputs.keys()))
