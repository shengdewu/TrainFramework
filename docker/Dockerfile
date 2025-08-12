FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND noninteractive

# -- change apt sources
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY docker/sources.list /etc/apt/

# -- install some tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common build-essential sudo ca-certificates \
    curl unzip wget git vim htop net-tools openssh-client libgl1 libglx-mesa0 python3-pip && \
    apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

COPY docker/requirements.txt /tmp/
RUN pip3 install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install --no-cache-dir Cython wheel && \
    pip3 install -r /tmp/requirements.txt && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache


# -- install locales && set timezone
RUN apt-get update -y && apt-get install -y --no-install-recommends tzdata locales locales-all && \
	locale-gen en_US.UTF-8 && echo "Asia/Shanghai" > /etc/timezone && \
	rm /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
	apt-get autoclean && apt-get -y clean && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
