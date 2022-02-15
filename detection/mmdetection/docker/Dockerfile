FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN  apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx openslide-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mmcv==0.6.2
# ==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu112/torch1.9.0/index.html

# Install mmdetection

RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN pip install --no-cache-dir -e .
RUN pip install openslide-python
RUN python setup.py develop
WORKDIR /work
