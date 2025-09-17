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

import sys
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, List, Optional, Type, cast, Tuple
import time
import threading
import websockets
import asyncio

# import aiohttp

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import datetime
import json
import uuid

import requests
import websocket
from requests.adapters import HTTPAdapter, Retry
from websocket import WebSocket

from .base import JupyterConnectionInfo
from datawiseagent.common.log import logger
from enum import Enum

class JupyterClient:
    """A client for managing and interacting with a Jupyter Kernel Gateway server.

    This class communicates with a Jupyter Kernel Gateway server via HTTP API,
    providing functionality to list kernel specs, manage running kernels, and
    start or delete specific kernels. Each instance uses the provided connection
    information to establish a session with the server and make authorized requests.

    Attributes:
        _connection_info (JupyterConnectionInfo): Connection information for the
            Jupyter Gateway, including host, port, and authentication token.
        _session (requests.Session): Session object to manage HTTP connections
            and retry logic.

    Methods:
        list_kernel_specs() -> Dict[str, Dict[str, str]]:
            List available kernel specifications on the server.

        list_kernels() -> List[Dict[str, str]]:
            List currently running kernels on the server.

        start_kernel(kernel_spec_name: str) -> str:
            Start a new kernel of the specified type and return the kernel ID.

        delete_kernel(kernel_id: str) -> None:
            Delete a running kernel by ID, terminating its process.

        restart_kernel(kernel_id: str) -> None:
            Restart a running kernel by ID.

        get_kernel_client(kernel_id: str) -> JupyterKernelClient:
            Create a WebSocket client for communicating with a specific kernel instance.
    """
    def __init__(self, connection_info: JupyterConnectionInfo):
        """(Experimental) A client for communicating with a Jupyter gateway server.

        Args:
            connection_info (JupyterConnectionInfo): Connection information
        """
        self._connection_info = connection_info

        # self._session = aiohttp.ClientSession()
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

    def _get_headers(self) -> Dict[str, str]:
        if self._connection_info.token is None:
            return {}
        return {"Authorization": f"token {self._connection_info.token}"}

    def _get_api_base_url(self) -> str:
        protocol = "https" if self._connection_info.use_https else "http"
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"{protocol}://{self._connection_info.host}{port}"

    def _get_ws_base_url(self) -> str:
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"ws://{self._connection_info.host}{port}"

    def list_kernel_specs(self) -> Dict[str, Dict[str, str]]:
        response = self._session.get(
            f"{self._get_api_base_url()}/api/kernelspecs", headers=self._get_headers()
        )

        # `cast`` is used to signal to the type checker that the value has the designated type but don't check anything at runtime.
        return cast(Dict[str, Dict[str, str]], response.json())

    def list_kernels(self) -> List[Dict[str, str]]:
        response = self._session.get(
            f"{self._get_api_base_url()}/api/kernels", headers=self._get_headers()
        )
        return cast(List[Dict[str, str]], response.json())

    def start_kernel(self, kernel_spec_name: str) -> str:
        """Start a new kernel.

        Args:
            kernel_spec_name (str): Name of the kernel spec to start

        Returns:
            str: ID of the started kernel
        """

        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels",
            headers=self._get_headers(),
            json={"name": kernel_spec_name},
        )
        return cast(str, response.json()["id"])

    def is_kernel_running(self, kernel_id: str) -> bool:
        response = self._session.get(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}",
            headers=self._get_headers(),
        )
        if 200 <= response.status_code < 300:
            return True
        elif response.status_code == 404:
            return False
        else:
            response.raise_for_status()

    def delete_kernel(self, kernel_id: str) -> None:
        response = self._session.delete(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}",
            headers=self._get_headers(),
        )
        # Check HTTP status code. Raises HTTPError, if one occurred.
        response.raise_for_status()

    def restart_kernel(self, kernel_id: str) -> None:
        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/restart",
            headers=self._get_headers(),
        )
        response.raise_for_status()

    def interrupt_kernel(self, kernel_id: str) -> None:
        """Interrupt a running kernel by sending an interrupt request.

        Args:
            kernel_id (str): The ID of the kernel to interrupt.

        Raises:
            requests.HTTPError: If the interrupt request fails.
        """
        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/interrupt",
            headers=self._get_headers(),
        )
        if response.status_code == 204:
            logger.info(f"Kernel {kernel_id} interrupted successfully.")
        else:
            logger.error(
                f"Failed to interrupt kernel {kernel_id}. Status code: {response.status_code}, Response: {response.text}"
            )
            response.raise_for_status()

    async def get_kernel_client(self, kernel_id: str) -> JupyterKernelClient:
        # Upgrades the connection to a websocket connection.
        # The /api/kernels/{kernel_id}/channels resource multiplexes the Jupyter kernel messaging protocol over a single Websocket connection.
        """
        ws_req = HTTPRequest(url='{}/api/kernels/{}/channels'.format(
                base_ws_url,
                url_escape(kernel_id)
            ),
            auth_username='fakeuser',
            auth_password='fakepass'
        )
        ws = yield websocket_connect(ws_req)
        print('Connected to kernel websocket')
        """
        ws_url = f"{self._get_ws_base_url()}/api/kernels/{kernel_id}/channels"
        # ws = websocket.create_connection(ws_url, header=self._get_headers())
        # return JupyterKernelClient(ws)
        return await JupyterKernelClient.create(
            ws_url, self._get_headers(), self, kernel_id
        )


class JupyterKernelClient:
    """A client for communicating with an individual Jupyter kernel via WebSocket.

    This class provides methods to interact with a specific Jupyter kernel,
    including sending code execution requests, receiving output, and handling
    execution states. It uses a WebSocket connection to maintain a direct
    communication channel with the kernel.

    Inner Class:
        ExecutionResult: Represents the result of a code execution request,
            including status, textual output, and MIME-type data output.

    Attributes:
        _session_id (str): Unique session identifier for the current client session.
        _websocket (WebSocket): WebSocket connection to the kernel instance.

    Methods:
        stop() -> None:
            Close the WebSocket connection.

        wait_for_ready(timeout_seconds: Optional[float] = None) -> bool:
            Check if the kernel is ready to receive execution requests.

        execute(code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
            Execute a code block on the kernel and return the result, handling output,
            display data, and errors.
    """

    @dataclass
    class ExecutionResult:
        @dataclass
        class DataItem:
            mime_type: str
            data: str

        is_ok: bool
        output: str
        data_items: List[DataItem]

    def __init__(
        self,
        ws_url: str,
        headers: Dict[str, str],
        jupyter_client: JupyterClient,
        kernel_id: str,
    ):
        self._session_id: str = uuid.uuid4().hex
        self._ws_url = ws_url
        self._headers = headers
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None

        self.jupyter_client = jupyter_client
        self.kernel_id = kernel_id

        self._stop_event = asyncio.Event()  # Event to control thread termination
        # self._connect()
        # self._start_heartbeat()

    @classmethod
    async def create(
        cls,
        ws_url: str,
        headers: Dict[str, str],
        jupyter_client: JupyterClient,
        kernel_id: str,
    ):
        self = cls(ws_url, headers, jupyter_client, kernel_id)
        await self._connect()
        # self._start_heartbeat()

        return self

    async def _connect_deprecated(self):
        """
        Establishes a WebSocket connection to the Jupyter kernel with retry logic.

        This method attempts to create a WebSocket connection to the specified kernel,
        retrying up to a maximum number of times if the connection fails. After each
        failure, it checks if the kernel is still running. If the kernel is no longer
        running, a RuntimeError is raised, halting further attempts. If the connection
        fails `MAX_RETRIES` times, it raises the most recent exception.

        Attributes:
            retry_count (int): Tracks the number of connection attempts.
            start_time (float): Records the time when the connection attempts started.
            MAX_RETRIES (int): Maximum allowed retry attempts for the WebSocket connection.

        Raises:
            RuntimeError: If the kernel is not running, stopping further attempts.
            Exception: If the connection fails after `MAX_RETRIES` attempts.

        Logging:
            - Logs a debug message on successful WebSocket connection, including retry count and time taken.
            - Logs an error message with exception details if a connection attempt fails.
            - Logs a warning if the kernel is found to be shut down during a retry.

        """
        MAX_RETRIES = 10
        retry_count = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                self._websocket = await websockets.connect(
                    self._ws_url,
                    extra_headers=self._headers,
                    ping_interval=None,  # 我们自己实现心跳机制
                    close_timeout=10,
                    # max_size=10 * 1024 * 1024,  # 10MB
                )
                # self._ws_url, header=self._headers, timeout=10
                # )
                logger.debug(
                    f"WebSocket connected after {retry_count} retries, took {time.time() - start_time:.2f} seconds."
                )
                break
            except Exception as e:
                retry_count += 1

                # Check if the kernel is running before attempting to connect
                if not self.jupyter_client.is_kernel_running(self.kernel_id):
                    logger.warn(
                        f"Kernel {self.kernel_id} is not running after {retry_count} retries. Total time: {time.time() - start_time:.2f} seconds."
                    )
                    raise RuntimeError(f"Kernel {self.kernel_id} is not running.")

                logger.error(
                    f"Retry {retry_count}: WebSocket Connection failed due to {e}. Retrying in 5 seconds..."
                )

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5)
                    logger.debug("Stop event set; exiting reconnection loop.")
                    break
                except asyncio.TimeoutError:
                    pass

                # raise the exception is retry_count exceeds MAX_RETRIES
                if retry_count >= MAX_RETRIES:
                    logger.error(
                        f"Exceeded maximum retry attempts ({MAX_RETRIES}) for WebSocket connection to kernel {self.kernel_id}."
                    )
                    raise e
        if self._stop_event.is_set():
            logger.debug("Connection attempt stopped due to stop event.")

    def _start_heartbeat(self):
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_task = asyncio.create_task(self._send_heartbeat())
        # heartbeat RuntimeError will not be catched.
        # Only if RuntimeError happens when connecting to kernel before executing code, the RuntimeError could be catched.

    async def _send_heartbeat(self):
        while not self._stop_event.is_set():  # check the stop event
            # Wait for the heartbeat interval or until the stop event is set
            try:

                try:
                    # 等待 stop_event 在 30 秒内触发，如果触发则直接 break
                    # 如果在 30 秒内没有触发 stop_event，则抛出 TimeoutError 进行 ping。
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._heartbeat_interval
                    )
                    # 如果执行到这里，说明 stop_event 已被触发
                    break
                except asyncio.TimeoutError:
                    pass

                if self._websocket:
                    try:
                        await self._websocket.ping()
                        logger.debug("WebSocket Ping sent.")
                    except Exception as e:
                        logger.error(
                            f"WebSocket Heartbeat failed: {e}. Reconnecting..."
                        )
                        await self._reconnect()
                else:
                    logger.debug("WebSocket is not connected. Reconnecting...")
                    await self._reconnect()
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop the client by closing the WebSocket and stopping the heartbeat."""
        logger.debug(
            f"Stopping JupyterKernelClient of the kernel id {self.kernel_id}..."
        )
        self._stop_event.set()  # Signal the heartbeat thread to stop

        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.debug("JupyterKernelClient has been stopped.")

    async def _connect(self):
        """
        Establishes a WebSocket connection to the Jupyter kernel with retry logic.

        This method attempts to create a WebSocket connection to the specified kernel,
        retrying up to a maximum number of times if the connection fails. After each
        failure, it checks if the kernel is still running. If the kernel is no longer
        running, a RuntimeError is raised, halting further attempts. If the connection
        fails `MAX_RETRIES` times, it raises the most recent exception.

        Attributes:
            retry_count (int): Tracks the number of connection attempts.
            start_time (float): Records the time when the connection attempts started.
            MAX_RETRIES (int): Maximum allowed retry attempts for the WebSocket connection.

        Raises:
            RuntimeError: If the kernel is not running, stopping further attempts.
            Exception: If the connection fails after `MAX_RETRIES` attempts.

        Logging:
            - Logs a debug message on successful WebSocket connection, including retry count and time taken.
            - Logs an error message with exception details if a connection attempt fails.
            - Logs a warning if the kernel is found to be shut down during a retry.

        """
        MAX_RETRIES = 10
        retry_count = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                self._websocket = await websockets.connect(
                    self._ws_url,
                    extra_headers=self._headers,
                    ping_interval=30,  # 使用内置心跳机制
                    ping_timeout=10,
                    close_timeout=20,
                    # max_size=10 * 1024 * 1024,  # 10MB
                )
                # self._ws_url, header=self._headers, timeout=10
                # )
                logger.debug(
                    f"WebSocket connected after {retry_count} retries, took {time.time() - start_time:.2f} seconds."
                )
                break
            except Exception as e:
                retry_count += 1

                # Check if the kernel is running before attempting to connect
                if not self.jupyter_client.is_kernel_running(self.kernel_id):
                    logger.warn(
                        f"Kernel {self.kernel_id} is not running after {retry_count} retries. Total time: {time.time() - start_time:.2f} seconds."
                    )
                    raise RuntimeError(f"Kernel {self.kernel_id} is not running.")

                logger.error(
                    f"Retry {retry_count}: WebSocket Connection failed due to {e}. Retrying in 5 seconds..."
                )

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5)
                    logger.debug("Stop event set; exiting reconnection loop.")
                    break
                except asyncio.TimeoutError:
                    pass

                # raise the exception is retry_count exceeds MAX_RETRIES
                if retry_count >= MAX_RETRIES:
                    logger.error(
                        f"Exceeded maximum retry attempts ({MAX_RETRIES}) for WebSocket connection to kernel {self.kernel_id}."
                    )
                    raise e
        if self._stop_event.is_set():
            logger.debug("Connection attempt stopped due to stop event.")

    async def _reconnect(self):
        await self._connect()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()

    async def _send_message(
        self, *, content: Dict[str, Any], channel: str, message_type: str
    ) -> str:
        timestamp = datetime.datetime.now().isoformat()
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "username": "autogen",
                "version": "5.0",
                "session": self._session_id,
                "msg_id": message_id,
                "msg_type": message_type,
                "date": timestamp,
            },
            "parent_header": {},
            "channel": channel,
            "content": content,
            "metadata": {},
            "buffers": [],  # TODO: "buffers": [],
        }
        """
        https://jupyter-client.readthedocs.io/en/stable/messaging.html
        `channel`(str): Literal["shell", "iopub", "stdin", "control", "heartbeat"]
        `message_type`(str): Literal[
            # shell 
                ## Execute
                "execute_request", "execute_reply",
                ## Introspection
                "inspect_request", "inspect_reply",
                ## Completion
                "complete_request", "complete_reply",
                ## History
                "history_request", "history_reply",
                ## Code completeness
                "is_complete_request", "is_complete_reply",
                ## Connect
                "connect_request", "connect_reply",
                ## Comm info
                "comm_info_request", "comm_info_reply",
                ## Kernel info
                "kernel_info_request", "kernel_info_reply",
            # control
                ## Kernel shutdown
                "shutdown_request", "shutdown_reply",
                ## Kernel interrupt
                "interrupt_request", "interrupt_reply",
                ## Debug request
                "debug_request", "debug_reply",
            # iopub
                ## Streams(stdout, stderr, etc)
                "stream",
                ## Display Data
                "display_data",
                ## Update Display Data
                "update_display_data",
                ## Code inputs
                "execute_input",
                ## Execution results
                "execute_results",
                ## Execution errors
                "error",
                ## Kernel status
                "status",
                ## Clear output
                "clear_output",
                ## Debug event
                "debug_event",
            # stdin
                "input_request", "input_reply",
            
            # heartbeat
                "comm_open", "comm_close", "comm_msg"
                ]
        `parent_header`:
        When a message is the “result” of another message, such as a side-effect (output or status) or direct reply, the parent_header is a copy of the header of the message that “caused” the current message. _reply messages MUST have a parent_header, and side-effects typically have a parent. If there is no parent, an empty dict should be used. This parent is used by clients to route message handling to the right place, such as outputs to a cell.
        """
        try:
            await self._websocket.send(json.dumps(message))
            # .send_text(json.dumps(message))
        except (websockets.ConnectionClosed, BrokenPipeError) as e:
            logger.error(f"Send failed: {e}. Reconnecting and retrying...")
            await self._reconnect()
            await self._websocket.send(
                json.dumps(message)
            )  # Retry once after reconnect
        return message_id

    async def _receive_message(
        self, timeout_seconds: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        if self._websocket is None:
            logger.debug("WebSocket is not connected. Reconnecting...")
            await self._reconnect()

        try:
            if timeout_seconds is not None:
                message = await asyncio.wait_for(
                    self._websocket.recv(), timeout=timeout_seconds
                )
            else:
                message = await self._websocket.recv()
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            return cast(Dict[str, Any], json.loads(message))
        except asyncio.TimeoutError:
            logger.debug(
                f"No message received within timeout ({timeout_seconds} seconds)."
            )
            return None
        except websockets.exceptions.PayloadTooBig as e:
            logger.error(f"Message too big: {e}.")
            await self._reconnect()
            raise e
        except (
            websockets.ConnectionClosed,
            ConnectionResetError,
        ) as e:
            logger.error(f"Receive failed: {e}. Reconnecting...")
            await self._reconnect()
            return None
        except Exception as e:
            logger.error(f"Unexpected error during receive: {e}")
            return None

    async def wait_for_ready(self, timeout_seconds: Optional[float] = None) -> bool:
        message_id = await self._send_message(
            content={}, channel="shell", message_type="kernel_info_request"
        )
        while True:
            message = await self._receive_message(timeout_seconds)
            # This means we timed out with no new messages.
            if message is None:
                return False
            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "kernel_info_reply"
            ):
                return True

    async def execute(
        self, code: str, timeout_seconds: Optional[float] = None
    ) -> ExecutionResult:
        MAX_OUTPUT_CHARACTERS = 3000  # 最大输出字符数限制
        TRUNCATION_MESSAGE = (
            "\n\n[Output truncated due to exceeding the maximum allowed size.]"
        )

        logger.info(f"Executing code with timeout_seconds={timeout_seconds}")
        message_id = await self._send_message(
            content={
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            channel="shell",
            message_type="execute_request",
        )

        text_output: List[Tuple[TextType, Any]] = []
        data_output = []
        # signal to trigger code verification failure
        is_ok_tag = True
        """
        type: stderr, other
        text_output: list[tuple(output_type, output_str)]
        """

        class TextType(Enum):
            STDERR = "stderr"
            OTHER = "other"

        def remove_duplicate_lines(stderr_output: str):
            # deduplicate
            lines = stderr_output.splitlines()
            unique_lines = set(lines)
            return "\n".join(unique_lines)

        def truncate_content(content: str):
            # 强制执行最大输出大小限制
            if len(content) > MAX_OUTPUT_CHARACTERS:
                content = content[:MAX_OUTPUT_CHARACTERS] + TRUNCATION_MESSAGE
                logger.warn(
                    "Output truncated due to exceeding the maximum allowed size."
                )
                logger.debug(
                    f"compelete output exceeding the max length {MAX_OUTPUT_CHARACTERS}:{content}"
                )
            return content

        def concat_text_output(text_output: List[Tuple[TextType, Any]]):
            content = ""
            stderr_buffer = ""
            for text_type, text in text_output:
                if text_type == TextType.OTHER:
                    # clear stderr_buffer
                    if stderr_buffer:
                        stderr_content = remove_duplicate_lines(stderr_buffer)
                        if stderr_content:
                            content += "\n" + stderr_content
                        stderr_buffer = ""
                    content += "\n" + str(text)
                elif text_type == TextType.STDERR:
                    stderr_buffer += "\n" + str(text)
            if stderr_buffer:
                stderr_content = remove_duplicate_lines(stderr_buffer)
                if stderr_content:
                    content += "\n" + stderr_content
                stderr_buffer = ""

            return truncate_content(content)

        while True:
            logger.info(f"Timeout seconds: {timeout_seconds}")
            try:
                message = await self._receive_message(timeout_seconds)
            except websockets.exceptions.PayloadTooBig as e:
                logger.error(f"Execution failed due to message too big: {e}")
                return JupyterKernelClient.ExecutionResult(
                    is_ok=False,
                    output="[Error] Output is too large to display.",
                    data_items=[],
                )
            if message is None:
                # timeout for execution
                try:
                    self.jupyter_client.interrupt_kernel(self.kernel_id)
                except Exception as e:
                    logger.error(f"Failed to interrupt kernel {self.kernel_id}: {e}")

                return JupyterKernelClient.ExecutionResult(
                    is_ok=False,
                    output=f"ERROR: Timeout waiting for output from code block. The time limit for each code blcok is {timeout_seconds}s and you should optimize the efficiency or reduce the complexity of each code block for execution.",
                    data_items=[],
                )

            # Ignore messages that are not for this execution.
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue

            msg_type = message["msg_type"]
            content = message["content"]
            if msg_type in ["execute_result", "display_data"]:
                for data_type, data in content["data"].items():
                    if data_type == "text/plain":
                        text_output.append((TextType.OTHER, data))
                    elif data_type.startswith("image/") or data_type == "text/html":
                        data_output.append(
                            self.ExecutionResult.DataItem(
                                mime_type=data_type, data=data
                            )
                        )
                    else:
                        text_output.append((TextType.OTHER, json.dumps(data)))
            elif msg_type == "stream":

                # Filter some critical warning types to trigger code verification failure.
                # Warning types: https://docs.python.org/zh-cn/3/library/exceptions.html

                # Most Strict
                CRITICAL_WARNING_PATTERNS = [
                    r"DeprecationWarning",
                    r"RuntimeWarning",
                    r"ResourceWarning",
                    r"SyntaxWarning",
                    r"UserWarning",
                ]

                if (
                    "name" in content
                    and content["name"] == "stderr"
                    and "text" in content
                ):
                    # warning messages
                    stderr_msg = content["text"]

                    # filter specific warnings to debugging stage
                    if any(
                        pattern in stderr_msg for pattern in CRITICAL_WARNING_PATTERNS
                    ):

                        # is_ok_tag = False
                        is_ok_tag = True

                    # content["text"] = remove_duplicate_lines(stderr_msg)
                    # print(stderr_msg)
                    # import pdb
                    # pdb.set_trace()
                    # (TextType.OTHER,
                    text_output.append((TextType.STDERR, content["text"]))
                else:
                    text_output.append((TextType.OTHER, content["text"]))

            elif msg_type == "error":
                # Output is an error.
                # content["traceback"] is a list of traceback lines
                # the content of message type `stream` or `error` could include ANSI escape sequences

                error_traceback = "\n".join(content["traceback"])
                # combined_output = "\n".join([str(output) for output in text_output])
                combined_output = concat_text_output(text_output)

                combined_output += (
                    f"ERROR: {content['ename']}: {content['evalue']}\n{error_traceback}"
                )

                is_ok_tag = False
                return JupyterKernelClient.ExecutionResult(
                    is_ok=is_ok_tag,
                    output=truncate_content(combined_output),
                    data_items=data_output,
                )
            if msg_type == "status" and content["execution_state"] == "idle":
                break

        return JupyterKernelClient.ExecutionResult(
            is_ok=is_ok_tag,
            # output="\n".join([str(output) for output in text_output]),
            output=truncate_content(concat_text_output(text_output)),
            data_items=data_output,
        )
