#!/bin/bash -e

cd $(dirname $0)/docker
docker build -t cifar10-model-importer .
docker build -t cifar10-model-converter -f Dockerfile.caffe .

cd $(git rev-parse --show-toplevel)
docker run -ti -v "$(pwd):$(pwd)" cifar10-model-converter /work/convert_hdf5_to_caffemodel.py "$(pwd)/tools/docker/cifar10_m7_train_test.prototxt" "$(pwd)/3rdparty/ML-examples/cmsisnn-cifar10/models/cifar10_m7_iter_300000.caffemodel.h5" "$(pwd)/3rdparty/ML-examples/cmsisnn-cifar10/models/cifar10_m7_iter_300000.caffemodel"
docker run -ti -v "$(pwd):$(pwd)" cifar10-model-importer /bin/bash -c "cd /work && PYTHONPATH=$(pwd)/3rdparty/ML-examples/cmsisnn-cifar10 python3 convert_cifar10_example_model.py --prototxt=$(pwd)/tools/docker/cifar10_m7_train_test.prototxt --caffemodel=$(pwd)/3rdparty/ML-examples/cmsisnn-cifar10/models/cifar10_m7_iter_300000.caffemodel --dest-dir=$(pwd)/data"
