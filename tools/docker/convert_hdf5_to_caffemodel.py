#!/usr/bin/python

import os
import subprocess
import sys

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

network, caffemodel_h5, caffemodel = sys.argv[1:4]
pb = caffe_pb2.NetParameter()
with open(network) as network_f:
  text_format.Merge(network_f.read().replace('/home/ubuntu/caffe', '/opt/caffe'), pb)

pb.layer[0].type = 'Input'
input_shape = caffe_pb2.BlobShape()
input_shape.dim.extend([1, 3, 32, 32])
pb.layer[0].input_param.shape.extend([input_shape])

rewritten_network = '{}.rewritten'.format(network)
with open(rewritten_network, 'w') as f:
  f.write(text_format.MessageToString(pb))

net = caffe.Net(rewritten_network, caffemodel_h5, caffe.TEST)
print 'saving'
net.save(caffemodel)
