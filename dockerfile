
FROM tensorflow/tensorflow:1.15.4-gpu-py3-jupyter

RUN apt-get update && \
      apt-get -y install sudo

RUN DEBIAN_FRONTEND="noninteractive"  apt-get install -y  protobuf-compiler python-pil python-lxml python-tk

RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user jupyter
RUN pip install --user matplotlib
RUN pip install --user tf_slim

RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user pillow
RUN pip install --user lxml
RUN pip install --user jupyter
RUN pip install --user matplotlib
RUN pip install --user tf_slim

RUN pip install pandas 

RUN pip install --user pycocotools

RUN mkdir -p /workspace/tensorflow/models/research
RUN cd /workspace/tensorflow/models/research

WORKDIR /
RUN apt-get -y install git && git clone https://github.com/tensorflow/tensorflow.git --branch r1.15 --single-branch --depth 1

EXPOSE 8888
EXPOSE 6006
WORKDIR /workspace/AudioRecognition

ENV PYTHONPATH "${PYTHONPATH}:/workspace/tensorflow/models/research:/workspace/tensorflow/models/research/slim"


