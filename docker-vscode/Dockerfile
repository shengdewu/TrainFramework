# ubuntu 20.04

FROM codercom/code-server:4.103.2-ubuntu

ENV DEBIAN_FRONTEND noninteractive

USER root

# -- change apt sources
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY sources.list /etc/apt/sources.list

# 设置时区
RUN apt-get update -y && apt-get install -y --no-install-recommends tzdata locales && \
	locale-gen en_US.UTF-8 && echo "Asia/Shanghai" > /etc/timezone && \
	rm /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
	apt-get autoclean && apt-get -y clean && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

#安装依赖
RUN apt-get update -y && apt-get install -y  curl unzip wget git vim htop net-tools openssh-client openssh-server make \
       wget openssl build-essential && \
    apt-get install -y pciutils  libblas-dev gfortran libblas3 \
      libopenblas-dev  dkms udev systemd libncurses5-dev libgdbm-dev liblzma-dev libz-dev libffi-dev \
      tk-dev libc6-dev zlib1g zlib1g-dev checkinstall libdb-dev libexpat1-dev \
      libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libtinfo-dev libbz2-dev \
      libgl1-mesa-glx python3-dev && \
    apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

#安装python
ARG PYTHON_VERSION=3.11.6

COPY Python-${PYTHON_VERSION}.tgz /tmp/
RUN cd /tmp/ && tar -xvzf Python-${PYTHON_VERSION}.tgz && cd /tmp/Python-${PYTHON_VERSION} \
    && zsh ./configure --prefix=/usr/local/python${PYTHON_VERSION} --enable-optimizations --with-ensurepip \
    && make -j4 && make install

ENV PATH="/usr/local/python${PYTHON_VERSION}/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/python${PYTHON_VERSION}/lib:$LD_LIBRARY_PATH"


#安装依赖
# https://mirrors.aliyun.com/pytorch-wheels/cu128 可以替换 https://download.pytorch.org/whl/cu128
RUN python3 --version && pip3 --version && pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    python3 -m pip install --upgrade pip && \
    pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
#分开, 防止更改requirements.txt 重新安装 torch
RUN pip3 install -r /tmp/requirements.txt && rm -rf /tmp/*

RUN sed -i 's/#Port 22/Port 8822/' /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo 'root:123456' | chpasswd && \
    cat /etc/ssh/sshd_config

COPY run.sh /opt/run.sh
RUN chmod u+x /opt/run.sh

RUN ln -s /usr/local/python${PYTHON_VERSION}/bin/pip3.11 /usr/local/bin/pip3.11 && \
    ln -s /usr/local/python${PYTHON_VERSION}/bin/python3.11 /usr/local/bin/python3.11

WORKDIR /home/workspace

ENTRYPOINT ["/opt/run.sh"]
