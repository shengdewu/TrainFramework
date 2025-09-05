# ubuntu 20.04

FROM codercom/code-server:4.103.2-ubuntu

ENV DEBIAN_FRONTEND=noninteractive

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
ARG PYTHON_MAJOR_VER=3.11
ARG PYTHON_MINOR_VER=6
ARG PYTHON_VERSION=${PYTHON_MAJOR_VER}.${PYTHON_MINOR_VER}

COPY Python-${PYTHON_VERSION}.tgz /tmp/
RUN cd /tmp/ && tar -xvzf Python-${PYTHON_VERSION}.tgz && cd /tmp/Python-${PYTHON_VERSION} \
    && ./configure --prefix=/usr/local/python${PYTHON_VERSION} --enable-optimizations --with-ensurepip \
    && make -j4 && make install && \
    ln -s /usr/local/python${PYTHON_VERSION}/bin/pip${PYTHON_MAJOR_VER} /usr/local/bin/pip${PYTHON_MAJOR_VER} && \
    ln -s /usr/local/python${PYTHON_VERSION}/bin/python${PYTHON_MAJOR_VER} /usr/local/bin/python${PYTHON_MAJOR_VER}


#安装依赖
# https://mirrors.aliyun.com/pytorch-wheels/cu128 可以替换 https://download.pytorch.org/whl/cu128
RUN python${PYTHON_MAJOR_VER} --version && pip${PYTHON_MAJOR_VER} --version && pip${PYTHON_MAJOR_VER} config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    python${PYTHON_MAJOR_VER} -m pip install --upgrade pip && \
    pip${PYTHON_MAJOR_VER} install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
#分开, 防止更改requirements.txt 重新安装 torch
RUN pip${PYTHON_MAJOR_VER} install --no-cache-dir -r /tmp/requirements.txt && rm -rf /tmp/*


WORKDIR /workspace

RUN sed -i 's/#Port 22/Port 8822/' /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo 'root:123456' | chpasswd && \
    cat /etc/ssh/sshd_config && \
    # ssh 登陆目录
    sed -i '$a cd /workspace' /root/.bashrc && \
    cat /root/.bashrc


COPY run.sh /opt/run.sh
RUN chmod u+x /opt/run.sh

ENTRYPOINT ["/opt/run.sh"]
