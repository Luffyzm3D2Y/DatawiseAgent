"""
class FileInfo:
    pass
    

class FileSystem:
    pass

    
"""

from __future__ import annotations
from pydantic import BaseModel, field_validator, ValidationError
from pathlib import Path
import uuid
import os
from typing import Literal, Union

from datawiseagent.common.utils import Singleton
from datawiseagent.common.log import logger


class FileInfo(BaseModel):
    filename: str
    # relative path to the working directory
    path: Path
    """
    # "image/*"
    "image/png"
    "text/html"
    # "text/csv"
    "text/plain"

    """
    mime_type: Union[
        Literal[
            "text/plain",
            "text/csv",
            "image/png",
            "application/json",
            "application/x-ndjson",
        ],
        str,
    ]

    @classmethod
    def extract_fileinfo(cls, path: Path | str) -> FileInfo:
        pass

    @field_validator("path")
    def check_relative_path(cls, value: Path):
        if value.is_absolute():
            raise ValueError("Path must be a relative path, not an absolute path.")
        return value


class Directory(BaseModel):
    dirname: str = ""
    # relative path to the working directory
    path: str = ""
    # TODO: create the concept of directory and file in the file system
    # TODO: implement the method of  __str__ to represent the status in string format at the root of this directory in `fs`


class FileSystem(metaclass=Singleton):
    """
    create workspace
    one workspace represent a directory in self.root_dir
    one workspace includes a log of sessions

    Now all sessions are stored in `self.root_dir`

    TODO:
        * multiple workspace for multiple users

    """

    def __init__(self, root_dir: str | Path = Path("/mnt/data/")):
        self.root_dir: str | Path = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if any(self.root_dir.iterdir()):
            error_msg = f"Directory {self.root_dir} is not empty! The root directory of file system of datawise agent should be empty."
            logger.error(error_msg)
            # raise FileExistsError(error_msg)

        else:
            info_msg = f"The root directory {self.root_dir} of file system in datawise agent is created successfully!"
            logger.info(info_msg)

    def fetch_filesystem_status(self, session_id: uuid.UUID) -> str:
        # TODO
        content = self.list_directory_tree(self.root_dir / str(session_id))
        if content.strip() == "":
            return "The workspace is empty now."
        return content

    # session management
    def create_session_env(self, session_id: uuid.UUID) -> bool:
        try:
            session_path = self.root_dir / str(session_id)
            session_path.mkdir(parents=True, exist_ok=True)
            info_msg = f"The workspace of session {session_id} is created successfully at the path of {session_path}.\nThe directory tree is displayed below:\n{self.list_directory_tree(session_path)}"
            logger.info(info_msg)
            return True
        except Exception as e:
            error_msg = f"Fail to create the workspace of session {session_id}!"
            logger.error(f"{e(error_msg)}")
            raise

    def list_directory_tree(self, path: Path, level: int = 0) -> str:
        # 存储目录树的内容
        content = []
        items = sorted(path.iterdir())  # 遍历目录中的所有文件和文件夹
        indent = "    " * level  # 用缩进表示树结构

        for item in items:
            if item.is_dir():
                content.append(f"{indent}📁 {item.name}")  # 如果是目录，递归显示子目录
                content.append(
                    self.list_directory_tree(item, level + 1)
                )  # 递归列出子目录
            else:
                content.append(f"{indent}📄 {item.name}")  # 如果是文件，直接添加文件名

        return "\n".join(content)  # 将列表转为字符串并返回


fs = FileSystem(root_dir=Path("/data1/yzm/datawiseagent/log/local"))
