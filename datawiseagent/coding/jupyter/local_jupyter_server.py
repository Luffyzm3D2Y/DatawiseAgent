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
import json
import secrets
import signal
import socket
import subprocess
import sys
from types import TracebackType
from typing import Optional, Type, Union, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from pathlib import Path
from .base import JupyterConnectable, JupyterConnectionInfo
from .jupyter_client import JupyterClient


class LocalJupyterServer(JupyterConnectable):
    """A class to manage a locally run Jupyter Kernel Gateway server.

    The LocalJupyterServer class is responsible for starting and managing a
    Jupyter Kernel Gateway server instance on the local machine. It includes
    options for setting up a token for authentication, configuring logging,
    and handling errors during the server startup process. The class also
    supports the context manager protocol, allowing the server to be managed
    with a `with` statement.

    Attributes:
        ip (str): IP address to bind the server to. Defaults to "127.0.0.1".
        port (Optional[int]): Port for the server. If None, an available port
            will be automatically chosen.
        token (Union[str, GenerateToken]): Token for server authentication.
            Defaults to GenerateToken, which generates a new token automatically.
        out_dir (str | Path): Working directory for the server.
        _subprocess (subprocess.Popen): Subprocess running the Jupyter Kernel
            Gateway server.

    ## Methods:
        stop() -> None:
            Stops the Jupyter Kernel Gateway server gracefully.

        connection_info -> JupyterConnectionInfo:
            Returns the connection information for the server, including IP,
            port, and token.

        get_client() -> JupyterClient:
            Returns a JupyterClient instance connected to the server, enabling
            kernel management and code execution.

        __enter__() -> Self:
            Context manager entry method. Returns the server instance.

        __exit__(exc_type, exc_val, exc_tb) -> None:
            Context manager exit method. Automatically stops the server.

    Example:
        LocalJupyterServer can be used to start and manage a Jupyter Kernel
        Gateway server locally. It supports automatic resource management via
        the context manager. Example usage:

        ```
        with LocalJupyterServer() as server:
            client = server.get_client()
            # Perform operations with the client
        ```

    Raises:
        ValueError: If running on Windows (due to known kernel gateway issue),
            or if Jupyter Kernel Gateway is not installed.
        ValueError: If the server fails to start, with detailed error information
            provided in the startup log.
    """
    class GenerateToken:
        pass

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: Optional[int] = None,
        token: Union[str, GenerateToken] = GenerateToken(),
        log_file: str = ".jupyter_gateway.log",
        log_level: str = "INFO",
        log_max_bytes: int = 1048576,
        log_backup_count: int = 0,
        out_dir: str | Path = Path("."),
    ):
        """Runs a Jupyter Kernel Gateway server locally.

        Args:
            ip (str, optional): IP address to bind to. Defaults to "127.0.0.1".
            port (Optional[int], optional): Port to use, if None it automatically selects a port. Defaults to None.
            token (Union[str, GenerateToken], optional): Token to use for Jupyter server. By default will generate a token. Using None will use no token for authentication. Defaults to GenerateToken().
            log_file (str, optional): File for Jupyter Kernel Gateway logs. Defaults to "jupyter_gateway.log".
            log_level (str, optional): Level for Jupyter Kernel Gateway logs. Defaults to "INFO".
            log_max_bytes (int, optional): Max logfile size. Defaults to 1048576.
            log_backup_count (int, optional): Number of backups for rotating log. Defaults to 3.
        """
        # Remove as soon as https://github.com/jupyter-server/kernel_gateway/issues/398 is fixed
        # if sys.platform == "win32":
        #    raise ValueError(
        #        "LocalJupyterServer is not supported on Windows due to kernelgateway bug."
        #    )

        # Check Jupyter gateway server is installed
        try:
            subprocess.run(
                [sys.executable, "-m", "jupyter", "kernelgateway", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                "Jupyter gateway server is not installed. Please install it with `pip install jupyter_kernel_gateway`."
            )

        self.ip = ip

        if isinstance(token, LocalJupyterServer.GenerateToken):
            token = secrets.token_hex(32)

        self.token = token
        self.out_dir = out_dir

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

        # Run Jupyter gateway server with detached subprocess
        args = [
            sys.executable,
            "-m",
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip",
            ip,
            "--KernelGatewayApp.auth_token",
            token,
            "--JupyterApp.answer_yes",
            "true",
            "--JupyterApp.logging_config",
            json.dumps(logging_config),
            "--JupyterWebsocketPersonality.list_kernels",
            "true",
        ]
        print(f"jupyter notebook working directory: {self.out_dir}")
        if port is not None:
            args.extend(["--KernelGatewayApp.port", str(port)])
            # args.extend(["--KernelGatewayApp.port_retries", "0"])
            args.extend(["--KernelGatewayApp.port_retries", "100"])
        self._subprocess = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.out_dir,  # set the working directory of the running kernel
        )

        # Satisfy mypy, we know this is not None because we passed PIPE
        assert self._subprocess.stderr is not None
        # Read stderr until we see "is available at" or the process has exited with an error
        stderr = ""
        while True:
            result = self._subprocess.poll()
            if result is not None:
                stderr += self._subprocess.stderr.read()
                raise ValueError(
                    f"Jupyter gateway server failed to start with exit code: {result}. stderr:\n{stderr}"
                )
            line = self._subprocess.stderr.readline()
            stderr += line

            if "ERROR:" in line:
                error_info = line.split("ERROR:")[1]
                raise ValueError(
                    f"Jupyter gateway server failed to start. {error_info}"
                )

            if "is available at" in line:
                # We need to extract what port it settled on
                # Example output:
                #   Jupyter Kernel Gateway 3.0.0 is available at http://127.0.0.1:8890
                if port is None:
                    port = int(line.split(":")[-1])
                self.port = port

                break

        # Poll the subprocess to check if it is still running
        result = self._subprocess.poll()
        if result is not None:
            raise ValueError(
                f"Jupyter gateway server failed to start. Please check the logs ({log_file}) for more information."
            )

        atexit.register(self.stop)

    def stop(self) -> None:
        if self._subprocess.poll() is None:
            if sys.platform == "win32":
                self._subprocess.send_signal(signal.CTRL_C_EVENT)
            else:
                # ctrl + C to terminate the subprocess
                self._subprocess.send_signal(signal.SIGINT)
            self._subprocess.wait()

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        return JupyterConnectionInfo(
            host=self.ip, use_https=False, port=self.port, token=self.token
        )

    def get_client(self) -> JupyterClient:
        return JupyterClient(self.connection_info)

    # Allow the class to be managed by a `with` statement by implementing `__enter__` and `__exit__` methods
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()
