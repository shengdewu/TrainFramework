#!/bin/bash

#  有问题立即退出
set -e

if [ -n "$ROOT_PASSWORD" ]; then
  echo "设置 root 密码 ..."
  echo "root:${ROOT_PASSWORD}" | chpasswd
fi

#启动ssh
service ssh start &

code-server --auth none --bind-addr 0.0.0.0:8080 /workspace