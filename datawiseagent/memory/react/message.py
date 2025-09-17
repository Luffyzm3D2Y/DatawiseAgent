# class Session:
# Chat History
# Session id
# Message (which could convert to LLM Message, just including one string input, and a list of files)
# LLM Message (mainly string, according to openai api protocol)
# workspace (a subset of file system)
#     pass


from pydantic import BaseModel, Field
from pathlib import Path

# import mimetypes
from typing import Literal, List, Union, Optional
import uuid

from .files import FileInfo

from datawiseagent.common.types import LLMResult


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    name: Literal["system", "user", "agent", "file system", "jupyter code executor"]
    text_content: str = ""
    files: list[FileInfo] = Field(default_factory=list)
    """
    User:
        text content
        a list of files
    Code Executor results
        text content
        a list of file paths (absolute path)
        Q: 
        1. 代码执行的富文本输出结果会被自动保存，例如html, png等。matplotlib作图可以保存。
        2. 如何让代码执行过程中保存的文件显式地出现在chat history中？
    File System
        text content
        a list of file paths (path in file system)
        2. 如何让代码执行过程中保存的文件显式地出现在chat history中？
    Agent response
        text content
    
    """

    def __init__(self, data: Optional[Union[LLMResult, dict]] = None, **kwargs):
        # 如果传入的是 LLMResult 类型
        if isinstance(data, LLMResult):
            # 执行 LLMResult 类型的初始化逻辑
            # TODO: debug
            if data.name not in [
                "system",
                "user",
                "agent",
                "file system",
                "jupyter code executor",
            ]:
                data.name = "agent"

            super().__init__(
                role=data.role, name=data.name, text_content=data.content, files=[]
            )
        # 如果传入的是字典
        elif isinstance(data, dict):
            # 执行正常的初始化逻辑
            super().__init__(**data)
        elif kwargs:
            super().__init__(**kwargs)
        else:
            raise TypeError("data must be either of type LLMResult or dict")

    def to_llmresult(self) -> LLMResult:
        if self.role == "user" and any(self.files):
            # TODO: when the uploaded file is image type, base64 encode it.
            paths = []
            for file in self.files:
                if file.mime_type == "image/png":
                    pass
                else:
                    paths.append(file.path)

            path_lines = "\n".join(map(str, paths))
            if path_lines.strip() == "":
                path_lines = "[]"
            llmresult = LLMResult(
                role=self.role,
                name=self.name,
                content=f"{self.text_content} Files uploaded to the following paths:\n {path_lines}",
            )
            return llmresult
        else:
            paths = []
            for file in self.files:
                paths.append(file.path)

            path_lines = "\n".join(map(str, paths))
            llmresult = LLMResult(
                role=self.role,
                name=self.name,
                content=f"{self.text_content} Files uploaded to the following paths:\n {path_lines}",
            )
            return llmresult


class Session(BaseModel):
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    chat_history: List[Message] = Field(default_factory=list)
    status: Literal["running", "terminated"] = "terminated"

    def to_llmresults(self) -> List[LLMResult]:
        llmresults = []
        for msg in self.chat_history:
            llmresults.append(msg.to_llmresult())

        return llmresults
