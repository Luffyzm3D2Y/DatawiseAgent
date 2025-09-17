# Copyright (c) 2024, Owners of https://github.com/Luffyzm3D2Y/DatawiseAgent
# SPDX-License-Identifier: Apache-2.0
#
# This file includes modifications based on code from:
#   https://github.com/microsoft/autogen
#   Copyright (c) Microsoft Corporation
#   SPDX-License-Identifier: MIT
#
# Substantial modifications and new features have been added.
from __future__ import annotations

import atexit
import io

# import logging
import json
import secrets
import sys
import uuid
from pathlib import Path
from types import TracebackType
from typing import Dict, Optional, Type, Union

import docker
import docker.errors
import time
import requests


from ..docker_commandline_code_executor import _wait_for_ready

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


from .base import JupyterConnectable, JupyterConnectionInfo
from .jupyter_client import JupyterClient

from datawiseagent.common.log import logger


def _wait_for_gateway(
    container, port, connection_info: JupyterConnectionInfo, timeout=60
):
    start_time = time.time()

    if connection_info.token is None:
        header = {}
    else:
        header = {"Authorization": f"token {connection_info.token}"}

    url = f"http://127.0.0.1:{port}/api/kernelspecs"

    while True:
        if time.time() - start_time > timeout:
            container.stop()
            raise TimeoutError("Timed out waiting for Jupyter gateway server to start.")

        try:
            response = requests.get(url, headers=header)
            if response.status_code == 200:
                logger.info("Jupyter gateway server is ready.")
                break
        except requests.exceptions.ConnectionError:
            pass  # 服务还未启动，继续等待

        container.reload()  # 更新容器状态
        if container.status != "running":
            logs = container.logs().decode("utf-8")
            raise ValueError(f"Container exited unexpectedly. Logs:\n{logs}")

        time.sleep(1)


class DockerJupyterServer(JupyterConnectable):
    DEFAULT_DOCKERFILE = """FROM quay.io/jupyter/docker-stacks-foundation

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER ${NB_UID}
RUN mamba install --yes jupyter_kernel_gateway ipykernel && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

ENV TOKEN="UNSET"
CMD python -m jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 \
    --KernelGatewayApp.port=8888 \
    --KernelGatewayApp.auth_token="${TOKEN}" \
    --JupyterApp.answer_yes=true \
    --JupyterWebsocketPersonality.list_kernels=true

EXPOSE 8888

WORKDIR "${HOME}"
"""

    class GenerateToken:
        pass

    def __init__(
        self,
        *,
        custom_image_name: Optional[str] = None,
        container_name: Optional[str] = None,
        auto_remove: bool = False,
        stop_container: bool = False,
        docker_env: Dict[str, str] = {},
        token: Union[str, GenerateToken] = GenerateToken(),
        dockerfile_path: Optional[Path] = None,
        log_file: str = "./.jupyter_gateway.log",
        log_level: str = "INFO",
        log_max_bytes: int = 1048576,
        log_backup_count: int = 0,
        out_dir: str | Path = Path("."),
        fresh_image: bool = False,
        use_proxy: bool = False,
        use_gpu: bool = False,
    ):
        """Start a Jupyter kernel gateway server in a Docker container.

        Args:
            custom_image_name (Optional[str], optional): Custom image to use. If this is None,
                then the bundled image will be built and used. The default image is based on
                quay.io/jupyter/docker-stacks-foundation and extended to include jupyter_kernel_gateway
            container_name (Optional[str], optional): Name of the container to start.
                A name will be generated if None.
            auto_remove (bool, optional): If true the Docker container will be deleted
                when it is stopped.
            stop_container (bool, optional): If true the container will be stopped,
                either by program exit or using the context manager
            docker_env (Dict[str, str], optional): Extra environment variables to pass
                to the running Docker container.
            token (Union[str, GenerateToken], optional): Token to use for authentication.
                If GenerateToken is used, a random token will be generated. Empty string
                will be unauthenticated.
        """
        self.out_dir = out_dir
        self.docker_root_dir = "/mnt"

        if container_name is None:
            container_name = f"jupyterkernelgateway-{uuid.uuid4()}"

        client = docker.from_env()
        here = Path(__file__).parent
        dockerfile_content = None
        if dockerfile_path and Path(dockerfile_path).is_file():
            dockerfile_content = Path(dockerfile_path).read_text(encoding="utf-8")
            logger.info(f"Using custom Dockerfile at: {dockerfile_path}")
        elif (here / "default_jupyter_server.dockerfile").is_file():
            dockerfile_content = (
                here / "default_jupyter_server.dockerfile"
            ).read_text()
            logger.info(f"Using default Dockerfile found in script directory {here}.")
        else:
            dockerfile_content = self.DEFAULT_DOCKERFILE
            logger.info("No Dockerfile found. Using internal DEFAULT_DOCKERFILE.")

        if custom_image_name is None:
            # image_name = "jupyterkernelgateway"
            image_name = "my-jupyter-image"
            # Make sure the image exists
            try:
                if not fresh_image:
                    client.images.get(image_name)
                    logger.info(f"Image {image_name} exists.")
                else:
                    dockerfile = io.BytesIO(dockerfile_content.encode("utf-8"))
                    logger.info(
                        f"Image {image_name} found, but let's refresh it. Rebuilding it now."
                    )
                    client.images.build(path=here, fileobj=dockerfile, tag=image_name)
                    logger.info(f"Image {image_name} built successfully.")

            except docker.errors.ImageNotFound:
                # Build the image
                # Get this script directory
                dockerfile = io.BytesIO(dockerfile_content.encode("utf-8"))
                logger.info(f"Image {image_name} not found. Building it now.")
                client.images.build(path=here, fileobj=dockerfile, tag=image_name)
                logger.info(f"Image {image_name} built successfully.")
        else:
            image_name = custom_image_name
            # Check if the image exists
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                raise ValueError(f"Custom image {image_name} does not exist")

        if isinstance(token, DockerJupyterServer.GenerateToken):
            self._token = secrets.token_hex(32)
        else:
            self._token = token

        logging_config = {
            "handlers": {
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "filename": log_file,
                }
            },
            "loggers": {
                "KernelGatewayApp": {
                    "level": log_level,
                    "handlers": ["file", "console"],
                }
            },
        }

        # Run the container
        import os

        env = {
            "TOKEN": self._token,
            "LOGGING_CONFIG": str(logging_config),
            "KERNELGATEWAYAPP_IP": "0.0.0.0",
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", None),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", None),
        }
        if use_proxy:
            import os

            docker_env = {
                "HTTP_PROXY": os.environ.get("HOST_HTTP_PROXY", None),
                # 使用 host.docker.internal 指向主机代理(windows/macos支持，linux只能直接使用主机的ip)
                "HTTPS_PROXY": os.environ.get("HOST_HTTPS_PROXY", None),
                "NO_PROXY": os.environ.get("HOST_NO_PROXY", None),
                # host.docker.internal 无需走代理
            }
            env.update(docker_env)
        """
        "JUPYTER_GATEWAY_ARGS": " ".join(
                [
                    "--KernelGatewayApp.ip 0.0.0.0",  # Set to 0.0.0.0 to allow external access
                    f"--KernelGatewayApp.auth_token {self._token}",
                    "--JupyterApp.answer_yes true",
                    # f"--JupyterApp.logging_config" + " " + logging_config_json,
                    "--JupyterWebsocketPersonality.list_kernels true",
                ]
            ),"""
        env.update(docker_env)
        try:
            if use_gpu:
                # use GPU

                container = client.containers.run(
                    image_name,
                    detach=True,
                    auto_remove=auto_remove,
                    environment=env,
                    publish_all_ports=True,  # 随机分配端口映射
                    name=container_name,
                    working_dir=self.docker_root_dir,  # 容器内的工作目录
                    volumes={
                        str(self.out_dir.resolve()): {
                            "bind": self.docker_root_dir,  # 容器内目录
                            "mode": "rw",  # 读写权限
                        }
                    },
                    device_requests=[
                        docker.types.DeviceRequest(
                            driver="nvidia",  # 使用 NVIDIA 驱动
                            # count=1,  # 使用所有 GPU（如果只想使用某个 GPU，可以指定 count=1 并设置 `device_ids`）
                            capabilities=[["gpu"]],  # 指定这是一个 GPU 设备请求
                            device_ids=["0"],  # 只请求设备 ID 为 3 的 GPU
                        )
                    ],
                )
            else:
                container = client.containers.run(
                    image_name,
                    detach=True,
                    auto_remove=auto_remove,
                    environment=env,
                    publish_all_ports=True,  # 随机分配端口映射
                    name=container_name,
                    working_dir=self.docker_root_dir,  # 容器内的工作目录
                    volumes={
                        str(self.out_dir.resolve()): {
                            "bind": self.docker_root_dir,  # 容器内目录
                            "mode": "rw",  # 读写权限
                        }
                    },
                )
        except docker.errors.APIError as e:
            raise ValueError(f"Failed to start Docker container: {e}")

        try:
            _wait_for_ready(container)
        except Exception as e:
            if auto_remove:
                container.remove(force=True)
            else:
                logger.error(f"Docker container failed to start: {e}")
                logger.info(f"Container '{container_name}' is kept for inspection.")
            raise ValueError(f"Docker container failed to start: {e}")

        container_ports = container.ports
        self._port = int(container_ports["8888/tcp"][0]["HostPort"])
        self._container_id = container.id

        try:
            _wait_for_gateway(
                container, port=self._port, connection_info=self.connection_info
            )
        except Exception as e:
            if auto_remove:
                container.remove(force=True)
            else:
                logger.error(f"Docker container failed to start: {e}")
                logger.info(f"Container '{container_name}' is kept for inspection.")
            raise ValueError(f"Docker container failed to start: {e}")

        def cleanup() -> None:
            try:
                inner_container = client.containers.get(container.id)
                inner_container.stop()
            except docker.errors.NotFound:
                pass

            atexit.unregister(cleanup)

        if stop_container:

            atexit.register(cleanup)

        self._cleanup_func = cleanup
        self._stop_container = stop_container

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        return JupyterConnectionInfo(
            host="127.0.0.1", use_https=False, port=self._port, token=self._token
        )

    def stop(self) -> None:
        self._cleanup_func()

    def get_client(self) -> JupyterClient:
        return JupyterClient(self.connection_info)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()
