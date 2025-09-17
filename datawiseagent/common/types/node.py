from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Annotated, Union
import uuid

from .cell import NotebookCell, MarkdownCell, CodeCell, CodeOutputCell, CellUnion
from datawiseagent.prompts.datawise import (
    AWAIT_TAG,
    END_STEP_TAG,
    END_DEBUG_TAG,
    DEBUG_FAIL_TAG,
    DEBUG_SUCCEED_TAG,
    ITERATE_ON_LAST_STEP,
    ADVANCE_TO_NEXT_STEP,
    FULFILL_INSTRUCTION,
)


class CompletionUsage(BaseModel):
    completion_tokens: int = 0
    """Number of tokens in the generated completion."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    total_tokens: int = 0
    """Total number of tokens used in the request (prompt + completion)."""


class Node(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    cells_generated: List[CellUnion] | CellUnion
    stage_name: Literal[
        "User Query",
        "Incremental Execution Stage",
        "Planning Stage",
        "Debugging Stage",
        "Post-debugging Stage",
    ]
    action_signal: Optional[str]

    parent_node_id: Optional[uuid.UUID] = None
    children_node_ids: Optional[List[uuid.UUID]] = Field(default=[])

    completion_usage: CompletionUsage = Field(default_factory=CompletionUsage)

    def to_string(self) -> str:

        content = f"### cells_generated:\n"
        if isinstance(self.cells_generated, NotebookCell):
            cells = [self.cells_generated]
        else:
            cells = self.cells_generated
        content += (
            "\n".join(
                [f"###### {str(type(item))}\n" + item.to_string() for item in cells]
            )
            + "\n"
        )
        # map(str, cells)

        content += f"### node id: {self.id}\n"
        content += f"### cells dynamic id: {str(list(map(id, cells)))}\n"

        if isinstance(self.parent_node_id, uuid.UUID):
            content += f"### parent node id:{self.parent_node_id}\n"

        if isinstance(self.children_node_ids, list):
            content += "### children node ids: " + str(self.children_node_ids) + "\n"
        return content


class UserNode(Node):
    stage_name: Literal["User Query"] = "User Query"
    action_signal: Optional[str] = AWAIT_TAG

    parent_node_id: Optional[uuid.UUID] = None
    children_node_ids: Optional[List[uuid.UUID]] = Field(default=[])


class ExecutionNode(Node):
    stage_name: Literal["Incremental Execution Stage"] = "Incremental Execution Stage"
    action_signal: str

    correct_cells: Optional[List[CellUnion]] = None
    cells_to_debug: Optional[List[CellUnion]] = None

    debugging_trace_ids: Optional[List[uuid.UUID]] = None
    post_debugging_result_id: Optional[uuid.UUID] = None

    def to_string(self) -> str:

        content = f"### cells_generated:\n"
        if isinstance(self.cells_generated, NotebookCell):
            cells = [self.cells_generated]
        else:
            cells = self.cells_generated
        content += (
            "\n".join(
                [f"###### {str(type(item))}\n" + item.to_string() for item in cells]
            )
            + "\n"
        )

        if self.correct_cells != None:
            content += (
                f"### correct_cells:\n"
                + "\n".join(
                    [
                        f"###### {str(type(item))}\n" + item.to_string()
                        for item in self.correct_cells
                    ]
                )
                + "\n"
            )

        if self.cells_to_debug != None:
            content += (
                f"### cells_to_debug:\n"
                + "\n".join(
                    [
                        f"###### {str(type(item))}\n" + item.to_string()
                        for item in self.cells_to_debug
                    ]
                )
                + "\n"
            )

        content += f"### node id: {self.id}\n"
        content += f"### cells dynamic id: {str(list(map(id, cells)))}\n"

        if self.correct_cells != None:
            content += f"### cell ids of correct_cells: {str(list(map(id, self.correct_cells)))}\n"

        if self.cells_to_debug != None:
            content += f"### cell ids of cells_to_debug: {str(list(map(id, self.cells_to_debug)))}\n"

        if self.debugging_trace_ids != None:
            content += f"### ids of Nodes in debugging trace: {str(self.debugging_trace_ids)}\n"

        if self.post_debugging_result_id != None:
            content += f"### Node id in post_debugging_result: {self.post_debugging_result_id}\n"

        if isinstance(self.parent_node_id, Node):
            content += f"### parent node id:{self.parent_node_id}\n"

        if isinstance(self.children_node_ids, list):
            content += "### children node ids: " + str(self.children_node_ids) + "\n"

        return content


class StepNode(ExecutionNode):
    stage_name: Literal["Planning Stage"] = "Planning Stage"
    # is_ok: bool = True
    # If is_ok is Flase, it's Phantom Step Node, which means the format is wrong.
    # Like it contains [STEP GOAL] with <Fulfill INSTRUCTION>
    # Like it exclude [STEP GOAL] with <Iterate on Current STEP> or <Advance to Next STEP>


class DebugNode(Node):
    stage_name: Literal["Debugging Stage"] = "Debugging Stage"
    parent_node_id: Optional[uuid.UUID] = None
    children_node_ids: Optional[List[uuid.UUID]] = None


class PostDebuggingNode(Node):
    stage_name: Literal["Post-debugging Stage"] = "Post-debugging Stage"
    parent_node_id: Optional[uuid.UUID] = None
    children_node_ids: Optional[List[uuid.UUID]] = None

NodeUnion = Annotated[
    Union[UserNode, ExecutionNode, StepNode, DebugNode, PostDebuggingNode],
    Field(discriminator="stage_name"),
]
