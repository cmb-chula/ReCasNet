FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev openslide-tools libgl1-mesa-glx tmux
RUN pip install cython openslide-python
RUN pip install matplotlib sklearn pandas opencv-contrib-python

