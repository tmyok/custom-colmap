ARG cuda_version=12.1.1
ARG cudnn_version=8
ARG ubuntu=22.04
FROM nvcr.io/nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel-ubuntu${ubuntu}

LABEL maintainer "Tomoya Okazaki"

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PIP_DEFAULT_TIMEOUT=100
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

RUN apt -y update && apt -y upgrade && \
    apt -y install --no-install-recommends \
        build-essential \
        cmake \
        git \
        graphviz \
        graphviz-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libceres-dev \
        libcgal-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libgl1-mesa-dev \
        libglew-dev \
        libglib2.0-0 \
        libgoogle-glog-dev \
        libgtest-dev \
        libmetis-dev \
        libqt5opengl5-dev \
        libsqlite3-dev \
        libsm6 \
        libx11-dev \
        libxext6 \
        libxrender1 \
        ninja-build \
        python3-dev \
        python3-pip \
        qtbase5-dev \
        vim \
        wget \
        zip \
        unzip && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    setuptools==70.0.0 \
        wheel==0.43.0 && \
    python3 -m pip install --no-cache-dir \
        torch==2.3.0+cu121 \
        torchvision==0.18.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip cache purge

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        cmake==3.29.5.1 \
        einops==0.8.0 \
        h5py==3.11.0 \
        kornia==0.7.2 \
        kornia-moons==0.2.9 \
        kornia-rs==0.1.3 \
        opencv-python==4.10.0.82 \
        timm==1.0.3 \
        tqdm==4.66.4 \
        transformers==4.41.2 && \
    python3 -m pip cache purge

# pycolmap
WORKDIR /home
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /home/colmap/
RUN git checkout dcfd14b300868c7eb1b360ebbca0dc21acd641b6
WORKDIR /home/colmap/build

# https://github.com/colmap/colmap/issues/1822
RUN cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86

RUN ninja install

WORKDIR /home/colmap/pycolmap
RUN python3 -m pip install .

# lightglue
WORKDIR /home
RUN git clone https://github.com/cvg/LightGlue.git
WORKDIR /home/LightGlue
RUN git checkout edb2b838efb2ecfe3f88097c5fad9887d95aedad
RUN python3 -m pip install -e .

WORKDIR /home/work