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
import warnings

model_name = "cifar10_cnn"
# Set to true if you want to train before inference
should_train = False
# Batch number after which you just don't care
stop_training_after = 1

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

            if stop_training_after > 0 and batch_num > stop_training_after:
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


def relay_micro_build(func, dev_config, params=None):
    """Create a graph runtime module with a micro device context from a Relay function.

    Parameters
    ----------
    func : relay.Function
        function to compile

    dev_config : TODO
        TODO

    params : dict
        input parameters that do not change during inference

    Return
    ------
    mod : tvm.module.Module
        graph runtime module for the target device
    """
    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(func, target="c", params=params)
    micro_mod = micro.create_micro_mod(c_mod, dev_config)
    ctx = tvm.micro_dev(0)
    mod = graph_runtime.create(graph, micro_mod, ctx)
    mod.set_input(**params)
    return mod


if should_train:
    cifar10_cnn = build_cifar10_cnn(gluon.nn.HybridSequential())
    cifar10_cnn.hybridize()
    train_model(cifar10_cnn)
    cifar10_cnn.export(model_name, epoch=1)

## Import model
#print("[Import Model]")
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    cifar10_cnn = gluon.nn.SymbolBlock.imports(f"{model_name}-symbol.json", ['data'], f"{model_name}-0001.params", ctx=ctx)
#
## Convert to Relay
#mod, params = relay.frontend.from_mxnet(
#    cifar10_cnn, shape={"data": (1, 3, 32, 32)})
#
#print("[Quantizing]")
#with relay.quantize.qconfig(skip_k_conv=0, round_for_shift=True):
#    mod = relay.module.Module.from_expr(relay.quantize.quantize(mod['main'], params))


#mod = relay.fromtext("""
#v0.0.4
#def @main(%data: Tensor[(1, 3, 32, 32), int8],
#    %hybridsequential0_conv0_weight: Tensor[(32, 3, 5, 5), int8],
#    %hybridsequential0_conv0_bias: Tensor[(32), int8],
#    %hybridsequential0_conv1_weight: Tensor[(32, 32, 5, 5), int8],
#    %hybridsequential0_conv1_bias: Tensor[(32), int8],
#    %hybridsequential0_conv2_weight: Tensor[(64, 32, 5, 5), int8],
#    %hybridsequential0_conv2_bias: Tensor[(64), int8],
#    %hybridsequential0_dense0_weight: Tensor[(10, 576), int8],
#    %hybridsequential0_dense0_bias: Tensor[(10), int8]) -> Tensor[(1, 10), int8] {
#  %0 = nn.conv2d(%data, %hybridsequential0_conv0_weight, padding=[2, 2], channels=32, kernel_size=[5, 5]) /* ty=Tensor[(1, 32, 32, 32), int8] */;
#  %1 = nn.bias_add(%0, %hybridsequential0_conv0_bias) /* ty=Tensor[(1, 32, 32, 32), int8] */;
#  %2 = nn.max_pool2d(%1, pool_size=[3, 3], strides=[2, 2]) /* ty=Tensor[(1, 32, 15, 15), int8] */;
#  %3 = nn.relu(%2) /* ty=Tensor[(1, 32, 15, 15), int8] */;
#  %4 = nn.conv2d(%3, %hybridsequential0_conv1_weight, padding=[2, 2], channels=32, kernel_size=[5, 5]) /* ty=Tensor[(1, 32, 15, 15), int8] */;
#  %5 = nn.bias_add(%4, %hybridsequential0_conv1_bias) /* ty=Tensor[(1, 32, 15, 15), int8] */;
#  %6 = nn.relu(%5) /* ty=Tensor[(1, 32, 15, 15), int8] */;
#  %7 = nn.avg_pool2d(%6, pool_size=[3, 3], strides=[2, 2], count_include_pad=True) /* ty=Tensor[(1, 32, 7, 7), int8] */;
#  %8 = nn.conv2d(%7, %hybridsequential0_conv2_weight, padding=[2, 2], channels=64, kernel_size=[5, 5]) /* ty=Tensor[(1, 64, 7, 7), int8] */;
#  %9 = nn.bias_add(%8, %hybridsequential0_conv2_bias) /* ty=Tensor[(1, 64, 7, 7), int8] */;
#  %10 = nn.relu(%9) /* ty=Tensor[(1, 64, 7, 7), int8] */;
#  %11 = nn.avg_pool2d(%10, pool_size=[3, 3], strides=[2, 2], count_include_pad=True) /* ty=Tensor[(1, 64, 3, 3), int8] */;
#  %12 = nn.batch_flatten(%11) /* ty=Tensor[(1, 3136), int8] */;
#  %13 = nn.dense(%12, %hybridsequential0_dense0_weight, units=10) /* ty=Tensor[(1, 10), int8] */;
#  nn.bias_add(%13, %hybridsequential0_dense0_bias, axis=-1) /* ty=Tensor[(1, 10), int8] */
#}
#""")
mod = relay.fromtext("""
v0.0.4
def @main(%data: Tensor[(1, 3, 32, 32), int8],
    %hybridsequential0_conv0_weight: Tensor[(32, 3, 5, 5), int8],
    %hybridsequential0_conv0_bias: Tensor[(32), int8],
    %hybridsequential0_conv1_weight: Tensor[(32, 32, 5, 5), int8],
    %hybridsequential0_conv1_bias: Tensor[(32), int8],
    %hybridsequential0_conv2_weight: Tensor[(64, 32, 5, 5), int8],
    %hybridsequential0_conv2_bias: Tensor[(64), int8],
    %hybridsequential0_dense0_weight: Tensor[(10, 576), int8],
    %hybridsequential0_dense0_bias: Tensor[(10), int8]) -> Tensor[(1, 10), int8] {
  %0 = nn.conv2d(%data, %hybridsequential0_conv0_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int8");
  %1 = nn.bias_add(%0, %hybridsequential0_conv0_bias);
  %2 = nn.max_pool2d(%1, pool_size=[3, 3], strides=[2, 2]);
  %3 = nn.relu(%2);
  %4 = nn.conv2d(%3, %hybridsequential0_conv1_weight, padding=[2, 2], channels=32, kernel_size=[5, 5], out_dtype="int8");
  %5 = nn.bias_add(%4, %hybridsequential0_conv1_bias);
  %6 = nn.relu(%5);
  %7 = nn.avg_pool2d(%6, pool_size=[3, 3], strides=[2, 2], count_include_pad=True);
  %8 = nn.conv2d(%7, %hybridsequential0_conv2_weight, padding=[2, 2], channels=64, kernel_size=[5, 5], out_dtype="int8");
  %9 = nn.bias_add(%8, %hybridsequential0_conv2_bias);
  %10 = nn.relu(%9);
  %11 = nn.avg_pool2d(%10, pool_size=[3, 3], strides=[2, 2], count_include_pad=True);
  %12 = nn.batch_flatten(%11);
  %13 = nn.dense(%12, %hybridsequential0_dense0_weight, units=10, out_dtype="int8");
  nn.bias_add(%13, %hybridsequential0_dense0_bias, axis=-1)
}
""")
# TODO: add clip, scale, and cast("int8") ops after each int16 out_dtype
# TODO: check exactly what the intermediate types of ARM's CMSIS-NN model are (where do they do casting?).

# generate random params
params = {}
for i, param in enumerate(mod['main'].params):  # skip the input param
    shape = list(map(lambda x: x.value, param.checked_type.shape))
    result = tvm.nd.array(np.random.randint(-35, 35, size=shape, dtype=np.int8), tvm.cpu(0))
    if i == 0:
        image = result
    else:
        params[param.name_hint] = result

## Grab an example
#image, label = get_sample_point()

# Begin a session
import time
print("[Initialization]")
DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
with micro.Session(DEV_CONFIG):
    # Build the function
    print("[Building]")
    start_time = time.time()
    graph_mod = relay_micro_build(mod['main'], DEV_CONFIG, params=params)
    end_time = time.time()
    print(f'  Build took {end_time - start_time} seconds')

    # Execute with `image` as the input.
    print("[Executing]")
    start_time = time.time()
    graph_mod.run(data=image)
    end_time = time.time()
    # Get output
    micro_output_np = graph_mod.get_output(0).asnumpy()
    print(f'  Model execution took {end_time - start_time} seconds')
    print(micro_output_np)

    ## Check prediction
    #print("[Moment of Truth]")
    #print(f'  Expected label: {label}')
    #prediction_idx = np.argmax(tvm_output.asnumpy()[0])
    #print(f'  Actual label: {prediction_idx}')


#from tvm.relay import create_executor
#main_gv = relay.GlobalVar('main')
#mod = relay.Module({main_gv: mod['main']})
#intrp = create_executor("debug", mod)
#f = intrp.evaluate(main_gv)
## need to maintain arg ordering
#args = [image] + [params[param.name_hint] for param in mod['main'].params[1:]]
#relay_output = f(*args)
#print(relay_output)

with tvm.build_config(disable_vectorize=True):
    graph, op_mod, params = relay.build(mod['main'], target="llvm", params=params)
graph_mod = graph_runtime.create(graph, op_mod, tvm.cpu(0))
graph_mod.set_input(**params)
graph_mod.run(data=image)
relay_output = graph_mod.get_output(0)

tvm.testing.assert_allclose(micro_output_np, relay_output_np)
