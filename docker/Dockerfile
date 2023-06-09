FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# -- change apt sources
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY sources.list /etc/apt/
COPY requirements.txt /home/

# -- install some tool
RUN rm /etc/apt/sources.list.d/cuda.list && \
    #rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common build-essential supervisor sudo ca-certificates \
    curl unzip wget git vim htop net-tools inetutils-ping openssh-client\
    gcc g++  libsm6 libxext6 libxrender-dev lsb-core byobu &&\
    apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

# -- install python and some lib
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libjpeg-dev libpng-dev libtiff-dev libgl1-mesa-glx libgoogle-glog0v5 libprotobuf10 \
    python3.6-dev python3-pip libpython3.6-dev && \
    apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

# -- install py3.6 package
RUN pip3 install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install --no-cache-dir Cython==0.29.21 scikit-build==0.11.1 wheel==0.35.1 && \
    pip3 install numpy==1.16.0 numba==0.51.2 scipy==1.4.1 pyyaml==5.1.2 easydict==1.9 json-tricks==3.13.5 pycrypto==2.6.1 &&\
    pip3 install bottle==0.12.18 paste==3.5.0 tqdm==4.54.1 matplotlib==3.3.3 munkres==1.1.4 ffmpeg==1.4 && \
    pip3 install --no-cache-dir Pillow==8.1.2 fvcore==0.1.5.post20210825 scikit-image==0.17.2 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

RUN pip3 install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install --no-cache-dir opencv-python==4.6.0.66 opencv-contrib-python==4.5.5.64 &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

RUN pip3 install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# -- install locales && set timezone
RUN apt-get update -y && apt-get install -y --no-install-recommends tzdata locales locales-all && \
	locale-gen en_US.UTF-8 && echo "Asia/Shanghai" > /etc/timezone && \
	rm /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
	apt-get autoclean && apt-get -y clean && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache


ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8