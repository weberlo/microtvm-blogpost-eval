FROM onnx/onnx-ecosystem:latest

RUN apt-get update
#RUN apt-get install -y python
#&& \
#    apt-get install -y software-properties-common && \
#    add-apt-repository ppa:deadsnakes/ppa && \
#    apt-get update && apt-get install -y python3.7

RUN mkdir /work
#RUN cd /work && curl -LJO https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip && unzip protoc-3.11.4-linux-x86_64.zip
#ENV PATH=/work/protoc/bin:$PATH

#RUN python3 -m ensurepip --upgrade
#COPY requirements.txt /work
#COPY constraints.txt /work
#RUN pip install -r /work/requirements.txt -c /work/constraints.txt
#RUN cd /usr/local/lib/python2.7/dist-packages/mxnet/tools/caffe_converter/ && make

COPY convert_cifar10_example_model.py /work
