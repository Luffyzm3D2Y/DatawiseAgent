#!/bin/bash

# 检查端口 8888 是否被占用
if lsof -i:8888; then
    echo "Port 8888 is occupied, killing the related process…"
    # 获取占用端口的进程 PID 并杀掉
    lsof -ti:8888 | xargs kill -9
else
    echo "Port 8888 is not occupied, continuing to start the service."
fi

python -m jupyter kernelgateway --KernelGatewayApp.ip ${KERNELGATEWAYAPP_IP} \
 --KernelGatewayApp.auth_token ${TOKEN} \
 --JupyterApp.answer_yes true \
 --JupyterWebsocketPersonality.list_kernels true \
 --KernelGatewayApp.port 8888 \
 --JupyterApp.logging_config "${LOGGING_CONFIG}"