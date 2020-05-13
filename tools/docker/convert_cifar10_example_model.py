import argparse
import os
import pickle
import subprocess
import sys

import coremltools
import onnxmltools


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--prototxt',
    required=True,
    help='path to arm example model')
  parser.add_argument(
    '--caffemodel',
    required=True,
    help='path to arm example model')

  parser.add_argument(
    '--dest-dir',
    required=True,
    help='path to place converted files')
  args = parser.parse_args()

  # Convert Caffe model to CoreML
  coreml_model = coremltools.converters.caffe.convert((args.caffemodel, args.prototxt))

  # Save CoreML model
  intermediate_model = os.path.join(args.dest_dir, 'intermediate.mlmodel')
  if not os.path.exists(args.dest_dir):
    os.makedirs(args.dest_dir)
  coreml_model.save(intermediate_model)

  # Load a Core ML model
  coreml_model = coremltools.utils.load_spec(intermediate_model)

  # Convert the Core ML model into ONNX
  onnx_model = onnxmltools.convert_coreml(coreml_model)

  # Save as protobuf
  onnxmltools.utils.save_model(onnx_model, os.path.join(args.dest_dir, 'cifar10.onnx'))


if __name__ == '__main__':
  main()
