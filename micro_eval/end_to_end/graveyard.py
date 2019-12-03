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


if SHOULD_TRAIN:
    cifar10_cnn = build_cifar10_cnn(gluon.nn.HybridSequential())
    cifar10_cnn.hybridize()
    train_model(cifar10_cnn)
    cifar10_cnn.export(MODEL_NAME, epoch=1)

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
