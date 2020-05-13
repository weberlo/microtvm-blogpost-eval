import argparse

from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay

from micro_eval import util
from micro_eval.util import model_util


def main():
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

  with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
    quantized = relay.quantize.quantize(mod, params, dataset=numpy_samples)

  print('quantized', quantized)
  with open(args.quantized_tvm_model, 'w') as f:
    f.write(tvm.ir.save_json(quantized))


if __name__ == '__main__':
  main()
