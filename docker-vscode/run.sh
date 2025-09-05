#!/bin/bash

#启动ssh
service ssh start &

code-server --auth none --bind-addr 0.0.0.0:8080 /workspace