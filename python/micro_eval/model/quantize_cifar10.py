import argparse

from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay
from tvm.relay import transform

from micro_eval import util
from micro_eval.util import model_util


def quantize(mod, params):
  with relay.quantize.qconfig(
      calibrate_mode='global_scale',
      global_scale=8.0,
      nbit_activation=8,
      dtype_activation="int8",
      skip_conv_layers=[],
      dtype_input="int8",
      dtype_weight="int8"):
    print('input:', mod)
    quantized = relay.quantize.quantize(mod, params) #, dataset=numpy_samples)
    print('output:', quantized)
#  with open(args.quantized_tvm_model, 'w') as f:
#    f.write(tvm.ir.save_json(quantized))


def quantize_cifar10():
  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx-model', default=f'{util.get_repo_root()}/data/cifar10.onnx',
                      help='path to unquantized cifar10 model')
  parser.add_argument('--num-samples', default=1024, type=int,
                      help='number of samples to use for data-aware quantization')
  parser.add_argument('--quantized-tvm-model',
                      default=f'{util.get_repo_root()}/data/quantized-cifar10.onnx',
                      help='path to write quantized cifar10 model')

  args = parser.parse_args()

  onnx_model = onnx.load(args.onnx_model)

  data_shape = util.LabelledShape(N=1, C=3, H=32, W=32, dtype='uint8')
  mod, params = relay.frontend.from_onnx(onnx_model, {"data": data_shape.shape})

  samples = model_util.get_sample_points(args.num_samples, data_shape.layout)
  numpy_samples = []
  for data in samples:
    numpy_samples.append({'data': data['data'].data})

  quantize(mod, params)


def quantize_test():
    # """Program:
    #    fn (%x : Tensor[(10, 10), float32]) {
    #      let %t1 = log(x);
    #      let %t2 = add(%t1, %x);
    #      %t1
    #    }
    # """
    # tp = relay.TensorType((10, 10), "float32")
    # x = relay.var("x", tp)
    # sb = relay.ScopeBuilder()
    # t1 = sb.let("t1", relay.log(x))
    # t2 = sb.let("t2", relay.add(t1, x))
    # sb.ret(t2)
    # f = relay.Function([x], sb.get())

    dshape = (1, 4, 16, 16)
    # TODO quantization shouldn't crash if a pre-quantized graph (already int8) is handed to it.
    dtype = 'float32'
    func_name = 'main'
    x = relay.var("x", shape=dshape, dtype=dtype)
    conv_expr = relay.nn.conv2d(
            x, relay.var("w"),
            kernel_size=(3, 3),
            padding=(1, 1),
            channels=4)
    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)

    # TODO crash is probs coming from non-const weights? add param for weight
    import numpy as np
    weight_ty = mod['main'].params[1].type_annotation
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    params = {
        'w': tvm.nd.array(np.random.uniform(0, 1, size=w_shape).astype(weight_ty.dtype), ctx=tvm.cpu(0))
    }
    quantize(mod, params)

if __name__ == '__main__':
#   quantize_cifar10()
  quantize_test()
