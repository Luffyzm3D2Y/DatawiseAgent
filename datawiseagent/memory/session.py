from pathlib import Path

# import mimetypes
from typing import Any, Literal, List, Union, Optional, TypedDict
import uuid
from uuid import UUID
from pydantic import BaseModel, ValidationError
import json

from .chat_history import BaseMemory, ChatHistoryMemory, CellHistoryMemory
from datawiseagent.common.types import (
    LLMResult,
    CodeCell,
    CodeOutputCell,
    MarkdownCell,
    NotebookCell,
    DatawiseAgentConfig,
    CodeExecutorConfig,
    UserInfo,
    SessionInfo,
)
from datawiseagent.coding import CodeExecutor, JupyterCodeExecutor
from datawiseagent.coding.jupyter import (
    DockerJupyterServer,
    LocalJupyterServer,
    JupyterCodeExecutor,
)

from datawiseagent.coding.base import CodeBlock, IPythonCodeResult
from datawiseagent.memory.files import FileSystem
from datawiseagent.common.config import global_config

SESSIONS_LOG_PATH = global_config["log"]["sessions_log_path"]

class Session:

    def __init__(
        self,
        chat_history: BaseMemory,
        user_root_dir: Path,
        user_id: UUID,
        session_name: Optional[str] = None,
    ):
        self.session_id: uuid.UUID = uuid.uuid4()
        self.session_name: Optional[str] = session_name

        self.chat_history: BaseMemory = chat_history

        """
        `/mnt` is the logical directory of session root directory.
        - /mnt
            - /input (user input)
            - /display (kernel output)
            - /working (agent output)
        """
        self.root_dir: Path = user_root_dir / str(self.session_id)
        self.display_dir: Path = self.root_dir / str("display")
        self.input_dir: Path = self.root_dir / str("input")
        self.system_dir: Path = self.root_dir / str("system")
        self.working_dir: Path = self.root_dir / str("working")

        # self.session_json_path
        if session_name:
            self.session_json_path: Path = (
                Path(SESSIONS_LOG_PATH)
                / "users"
                / str(user_id)
                / "sessions"
                / str(self.session_id)
                / f"{self.session_name}.json"
            )
        else:
            self.session_json_path: Path = (
                Path(SESSIONS_LOG_PATH)
                / "users"
                / str(user_id)
                / "sessions"
                / str(self.session_id)
                / f"session.json"
            )

        self.workspace: FileSystem = FileSystem(
            self.session_id,
            self.session_json_path,
            self.root_dir,
            self.display_dir,
            self.input_dir,
            self.working_dir,
            self.system_dir,
        )

        self.code_executor: Optional[CodeExecutor] = None

        self.agent_config: Optional[DatawiseAgentConfig,] = None
        self.code_executor_config: Optional[CodeExecutorConfig] = None

        self.current_user_query: Optional[str] = None

        """
        code_executor could be jupyter kernel or just python interpreter.
        """

    async def rerun_cells(self):
        # Restart the kernel and rerun all the cells
        await self.code_executor.restart()
        assert isinstance(self.chat_history, CellHistoryMemory)

        # TODO: recover the status of file system
        self.workspace.initialize([self.input_dir, self.system_dir])

        # Rerun all the code cells
        new_cells: list[NotebookCell] = []
        code_cells_stack: list[CodeCell] = []
        for cell in self.chat_history.cells:
            if isinstance(cell, MarkdownCell):
                new_cells.append(cell)
            elif isinstance(cell, CodeCell):
                code_cells_stack.append(cell)

                if cell.code_output is not None:
                    cell.code_output.code_result = None

                new_cells.append(cell)
            elif isinstance(cell, CodeOutputCell):
                if len(code_cells_stack) != 0:
                    for code_cell in code_cells_stack:
                        # TODO: rude way to judge which code block is to trigger timeout
                        if (
                            code_cell.code_output != None
                            and "ERROR: Timeout waiting for output from code block."
                            in code_cell.code_output.to_string()
                        ):

                            code_output = await self.code_executor.execute_code_blocks(
                                code_cell.to_code_block(), custom_timeout=0.1
                            )
                        else:
                            code_output = await self.code_executor.execute_code_blocks(
                                code_cell.to_code_block()
                            )

                        if code_cell.code_output is not None:
                            code_cell.code_output.update_code_result(code_output)
                        else:
                            code_cell.code_output = CodeOutputCell(
                                code_result=code_output
                            )

                        new_cells.append(code_cell.code_output)
                    code_cells_stack.clear()
        self.chat_history.cells = new_cells

    async def safe_execute_code_blocks(
        self, code_blocks: CodeBlock | List[CodeBlock]
    ) -> IPythonCodeResult:

        if isinstance(self.code_executor, JupyterCodeExecutor):
            # pay attention to jupyter gateway server and jupyter kernel
            try:
                await self.code_executor.check_jupyter_kernel_health()
            except RuntimeError as e:
                if "Kernel" in str(e):
                    # The kernel shutdown
                    # restart the kernel

                    await self.rerun_cells()
                else:
                    raise e

            return await self.code_executor.execute_code_blocks(code_blocks)

    def workspace_root_dir(self):
        assert isinstance(self.code_executor, JupyterCodeExecutor)
        if isinstance(self.code_executor.jupyter_server, DockerJupyterServer):
            workspace_root_dir = self.code_executor.jupyter_server.docker_root_dir
        elif isinstance(self.code_executor.jupyter_server, LocalJupyterServer):
            workspace_root_dir = self.code_executor.jupyter_server.out_dir

        return workspace_root_dir


class SessionContent(BaseModel):
    """
    Represents the content of a session, encapsulating all essential configurations,
    user details, session-specific metadata, and chat history. This model is designed
    to support serialization and deserialization for persistent storage.

    Attributes:
    ----------
    llm_config : Optional[dict]
        Configuration settings for the language model (e.g., model name, parameters).
        This is used to control the behavior of the LLM during the session.

    agent_config : Optional[DatawiseAgentConfig]
        Configuration specific to the datawise agent, defining behaviors like
        planning, execution, and debugging settings.

    code_executor_config : Optional[CodeExecutorConfig]
        Configuration for the code executor, which manages the execution environment
        (e.g., kernel types, timeouts).

    user_info : Optional[UserInfo]
        Information about the user associated with the session, including user-specific
        preferences or identifiers.

    session_info : Optional[SessionInfo]
        Metadata about the session, such as session name, ID.
        Useful for tracking and managing multiple sessions.

    chat_history : Optional[BaseMemory]
        Stores the chat history of the session, including user queries and agent responses.
        This can be used to reconstruct interactions or provide context for follow-up tasks.

    fs_session_root_dir : Optional[Path]
        The root directory path for the session's file system. This is where
        session-specific files, outputs, and logs are stored.
    """

    llm_config: Optional[dict] = None
    agent_config: Optional[DatawiseAgentConfig] = None
    code_executor_config: Optional[CodeExecutorConfig] = None

    user_info: Optional[UserInfo] = None
    session_info: Optional[SessionInfo] = None

    chat_history: Optional[CellHistoryMemory] = None

    fs_session_root_dir: Optional[Path] = None

    @classmethod
    def from_json(
        cls,
        obj: Any,
    ):
        if isinstance(obj, str):
            # 假设输入是 JSON 字符串
            try:
                data = json.loads(obj)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON data: {e}")
        elif isinstance(obj, dict):
            data = obj
        else:
            raise ValidationError("Input must be a JSON string or a dictionary.")
        # 处理 chat_history 字段，使用 CellHistoryMemory 的 from_json 方法
        if "chat_history" in data:
            if data["chat_history"] is not None:
                try:
                    data["chat_history"] = CellHistoryMemory.from_json(
                        data["chat_history"]
                    )
                except ValidationError as e:
                    raise ValidationError(f"chat_history 反序列化失败: {e}") from e

        # 现在，使用 Pydantic 的标准方法创建 SessionContent 实例
        try:
            return cls(**data)
        except ValidationError as e:
            raise ValidationError(f"SessionContent 反序列化失败: {e}") from e
