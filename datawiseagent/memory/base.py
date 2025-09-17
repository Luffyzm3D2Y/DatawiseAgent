from abc import abstractmethod
from typing import List, Union

from pydantic import BaseModel

from datawiseagent.common.types.llm import LLMResult
from datawiseagent.common.types.cell import NotebookCell
from datawiseagent.common.types.node import Node

Message = Union[LLMResult, NotebookCell, Node]

class BaseMemory(BaseModel):
    @abstractmethod
    def add_messages(self, messages: Message | List[Message]) -> None:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def to_messages(self) -> List[dict]:
        pass
