FROM nvcr.io/nvidia/tensorflow:21.02-tf2-py3

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev openslide-tools libgl1-mesa-glx
RUN pip install cython openslide-python 
RUN pip install matplotlib sklearn pandas opencv-contrib-python imgaug efficientnet 
RUN pip install tf-nightly

