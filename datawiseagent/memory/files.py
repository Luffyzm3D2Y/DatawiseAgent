from __future__ import annotations
from pydantic import BaseModel, field_validator, ValidationError
from pathlib import Path
import uuid
import os
from typing import Literal, Union
import subprocess

from datawiseagent.common.utils import Singleton
from datawiseagent.common.log import logger

class FileSystem:
    """
    Workspace in one session of one user.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        session_json_path: str | Path,
        session_root_dir: str | Path = "/mnt",
        session_display_dir: str | Path = "/mnt/display",
        session_input_dir: str | Path = "/mnt/input",
        session_working_dir: str | Path = "/mnt/working",
        session_system_dir: str | Path = "/mnt/system",
        rebuild: bool = False,
    ):
        self.session_json_path: Path = Path(session_json_path)
        self.root_dir: Path = Path(session_root_dir)
        self.display_dir: Path = Path(session_display_dir)
        self.input_dir: Path = Path(session_input_dir)
        self.working_dir: Path = Path(session_working_dir)
        self.system_dir: Path = Path(session_system_dir)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not rebuild and any(self.root_dir.iterdir()):
            error_msg = f"Directory {self.root_dir} is not empty! The root directory of session {session_id} should be empty."
            logger.error(error_msg)

            self.display_dir.mkdir(parents=True, exist_ok=True)
            self.input_dir.mkdir(parents=True, exist_ok=True)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self.system_dir.mkdir(parents=True, exist_ok=True)
        else:
            info_msg = f"The root directory {self.root_dir} of session {session_id} is created successfully!"
            logger.info(info_msg)

            self.display_dir.mkdir(parents=True, exist_ok=True)
            self.input_dir.mkdir(parents=True, exist_ok=True)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self.system_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Physical Root Dir: {self.root_dir}\n"
                + "Workspace Status:\n"
                + self.fetch_workspace_status()
            )

        if not self.session_json_path.exists():
            # Ensure the parent directories exist
            self.session_json_path.parent.mkdir(parents=True, exist_ok=True)

            # Create an empty JSON file
            with open(self.session_json_path, "w") as f:
                f.write("{}")  # Write an empty JSON object

    def initialize(
        self, maintained_dirs: list[str | Path] = ["/mnt/input", "/mnt/system"]
    ):
        resolved_maintained_dirs = []
        for one_dir in maintained_dirs:
            one_dir = Path(one_dir).resolve()
            resolved_maintained_dirs.append(one_dir)
            if not one_dir.exists():
                error_msg = f"Directory {one_dir} does not exist."
                logger.error(error_msg)
                return

        # assertation for test
        # assert_dir = Path(__file__).parent.parent.parent / "log"

        # assert self.root_dir.resolve().is_relative_to(
        #    assert_dir.resolve()
        # ), f"{self.root_dir} is not {assert_dir.resolve()}"

        for item in self.root_dir.iterdir():
            # 检查是否是隐藏文件或目录（以 . 开头）
            if item.name.startswith("."):
                logger.info(f"Skipping hidden item {item}.")
                continue  # 如果是隐藏文件或目录，跳过删除操作

            if item.resolve() not in resolved_maintained_dirs:
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                        logger.info(f"File {item} has been removed.")
                    elif item.is_dir():
                        # TODO: deletion operation is dangerous!

                        import shutil

                        shutil.rmtree(item)
                        logger.info(f"Directory {item} has been removed.")
                except PermissionError as e:
                    logger.warn(f"Permission denied while removing {item}: {e}")
                    logger.info(f"Attempting to remove {item} with sudo.")
                    try:
                        if item.is_file() or item.is_symlink():
                            subprocess.run(["sudo", "rm", "-f", str(item)], check=True)
                            logger.info(
                                f"File {item} has been forcefully removed with sudo."
                            )
                        elif item.is_dir():
                            subprocess.run(["sudo", "rm", "-rf", str(item)], check=True)
                            logger.info(
                                f"Directory {item} has been forcefully removed with sudo."
                            )
                    except subprocess.CalledProcessError as sudo_e:
                        logger.error(f"Failed to remove {item} with sudo: {sudo_e}")
                except Exception as e:
                    logger.error(f"Failed to remove {item}: {e}")

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)
        self.display_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialization completed. Only {str(maintained_dirs)} remains in {self.root_dir}."
        )

    def fetch_workspace_status(self) -> str:
        """
        Returns the status of the workspace as a string by using the `tree` command.

        tree -F ./
        """
        result = subprocess.run(
            # ["tree", "-F", "./"],
            ["tree", "-Fsh", "-I", "__pycache__|*.pyc", "./"],
            capture_output=True,
            text=True,
            cwd=self.root_dir,
        )

        if result.returncode == 0:
            return result.stdout
        else:
            return "Error fetching workspace status: " + result.stderr


if __name__ == "__main__":
    # test
    """
    put the below 2 lines of code on the top of the file:

    import sys
    sys.path.append("../../.")
    """
    import pdb

    pdb.set_trace()
    fs = FileSystem(
        uuid.uuid4(),
        "./test",
        "./test",
        "./test",
        "./test",
    )
    print(fs.fetch_workspace_status())
