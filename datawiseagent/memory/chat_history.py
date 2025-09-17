import copy
from string import Template
from typing import Dict, List, Optional, Tuple, Literal, Union, Any
import openai
from pydantic import Field, ValidationError, TypeAdapter
import uuid
import json

from datawiseagent.llms import OpenAIChat
from datawiseagent.llms.utils import count_message_tokens, count_string_tokens
from datawiseagent.common.types.llm import LLMResult
from datawiseagent.common.types.cell import (
    NotebookCell,
    MarkdownCell,
    CodeCell,
    CodeOutputCell,
    FormatType,
    StepCell,
    UserCell,
    CellUnion,
)
from datawiseagent.common.types.node import (
    Node,
    StepNode,
    ExecutionNode,
    DebugNode,
    PostDebuggingNode,
    UserNode,
    NodeUnion,
)
from datawiseagent.prompts.datawise import (
    END_DEBUG_TAG,
    DEBUG_FAIL_TAG,
    DEBUG_SUCCEED_TAG,
    AWAIT_TAG,
    END_STEP_TAG,
    END_DEBUG_TAG,
    DEBUG_FAIL_TAG,
    DEBUG_SUCCEED_TAG,
    ITERATE_ON_LAST_STEP,
    ADVANCE_TO_NEXT_STEP,
    FULFILL_INSTRUCTION,
    DEBUGGING_SYSTEM_PROMPT,
    POST_DEBUGGING_SYSTEM_PROMPT,
)
from datawiseagent.common.log import logger
from . import memory_registry
from .base import BaseMemory


@memory_registry.register("chat_history")
class ChatHistoryMemory(BaseMemory):
    messages: List[LLMResult] = Field(default=[])
    has_summary: bool = False
    max_summary_tlength: int = 500
    last_trimmed_index: int = 0
    summary: str = ""
    SUMMARIZATION_PROMPT: str = '''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
${summary}
"""

Latest Development:
"""
${new_events}
"""
'''

    def add_messages(self, messages: LLMResult | List[LLMResult]) -> None:
        if isinstance(messages, LLMResult):
            messages = [messages]
        for message in messages:
            self.messages.append(message)

    def to_string(self, add_sender_prefix: bool = False) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    (
                        f"[{message.sender}]: {message.content}"
                        if message.sender != ""
                        else message.content
                    )
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])

    async def to_messages(
        self,
        my_name: str = "",
        start_index: int = 0,
        max_summary_length: int = 0,
        max_send_token: int = 0,
        model: str = "gpt-3.5-turbo",
    ) -> List[dict]:
        if self.has_summary:
            start_index = self.last_trimmed_index

        messages = [
            message.to_openai_message() for message in self.messages[start_index:]
        ]

        # summary message
        if self.has_summary:
            """https://github.com/Significant-Gravitas/AutoGPT/blob/release-v0.4.7/autogpt/memory/message_history.py"""
            if max_summary_length == 0:
                max_summary_length = self.max_summary_tlength
            max_send_token -= max_summary_length
            prompt = []
            trimmed_history = add_history_upto_token_limit(
                prompt, messages, max_send_token, model
            )
            if trimmed_history:
                new_summary_msg, _ = await self.trim_messages(
                    list(prompt), model, messages
                )
                prompt.append(new_summary_msg)
            messages = prompt
        return messages

    def reset(self) -> None:
        self.messages = []

    async def trim_messages(
        self, current_message_chain: List[Dict], model: str, history: List[Dict]
    ) -> Tuple[Dict, List[Dict]]:
        new_messages_not_in_chain = [
            msg for msg in history if msg not in current_message_chain
        ]

        if not new_messages_not_in_chain:
            return self.summary_message(), []

        new_summary_message = await self.update_running_summary(
            new_events=new_messages_not_in_chain, model=model
        )

        last_message = new_messages_not_in_chain[-1]
        self.last_trimmed_index += history.index(last_message)

        return new_summary_message, new_messages_not_in_chain

    async def update_running_summary(
        self,
        new_events: List[Dict],
        model: str = "gpt-3.5-turbo",
        max_summary_length: Optional[int] = None,
    ) -> dict:
        if not new_events:
            return self.summary_message()
        if max_summary_length is None:
            max_summary_length = self.max_summary_tlength

        new_events = copy.deepcopy(new_events)

        # Replace "assistant" with "you". This produces much better first person past tense results.
        for event in new_events:
            if event["role"].lower() == "assistant":
                event["role"] = "you"

            elif event["role"].lower() == "system":
                event["role"] = "your computer"

            # Delete all user messages
            elif event["role"] == "user":
                new_events.remove(event)

        prompt_template_length = len(
            Template(self.SUMMARIZATION_PROMPT).safe_substitute(
                summary="", new_events=""
            )
        )
        max_input_tokens = OpenAIChat.send_token_limit(model) - max_summary_length
        summary_tlength = count_string_tokens(self.summary, model)
        batch: List[Dict] = []
        batch_tlength = 0

        for event in new_events:
            event_tlength = count_message_tokens(event, model)

            if (
                batch_tlength + event_tlength
                > max_input_tokens - prompt_template_length - summary_tlength
            ):
                await self._update_summary_with_batch(batch, model, max_summary_length)
                summary_tlength = count_string_tokens(self.summary, model)
                batch = [event]
                batch_tlength = event_tlength
            else:
                batch.append(event)
                batch_tlength += event_tlength

        if batch:
            await self._update_summary_with_batch(batch, model, max_summary_length)

        return self.summary_message()

    async def _update_summary_with_batch(
        self, new_events_batch: List[dict], model: str, max_summary_length: int
    ) -> None:
        prompt = Template(self.SUMMARIZATION_PROMPT).safe_substitute(
            summary=self.summary, new_events=new_events_batch
        )

        self.summary = await openai.ChatCompletion.acreate(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_summary_length,
            temperature=0.5,
        )["choices"][0]["message"]["content"]

    def summary_message(self) -> dict:
        return {
            "role": "system",
            "content": f"This reminds you of these events from your past: \n{self.summary}",
        }


def add_history_upto_token_limit(
    prompt: List[dict], history: List[dict], t_limit: int, model: str
) -> List[LLMResult]:
    limit_reached = False
    current_prompt_length = 0
    trimmed_messages: List[Dict] = []
    for message in history[::-1]:
        token_to_add = count_message_tokens(message, model)
        if current_prompt_length + token_to_add > t_limit:
            limit_reached = True

        if not limit_reached:
            prompt.insert(0, message)
            current_prompt_length += token_to_add
        else:
            trimmed_messages.insert(0, message)
    return trimmed_messages


@memory_registry.register("cell_history")
class CellHistoryMemory(BaseMemory):
    """
    Stores and manages the history of notebook cells and corresponding node trees.

    Attributes:
    - cells: A list of `NotebookCell` objects representing the sequence of "right" interactions.
    - node_trees: A list of root `UserNode` objects forming trees of agent interactions.

    Methods:
    - add_messages(messages): Adds one or more `NotebookCell` objects to the history.
    - to_string(add_sender_prefix): Converts the history of cells to a string representation.
    - reset(): Clears the cell history.
    - find_last_step(): Finds the last `StepCell` in the history.
    - clear_last_step(): Clears all cells after the last `StepCell`.
    - to_messages(format, insert_before_last_step_prompt): Converts cells to a list of OpenAI messages.
    - add_node_tree(user_node): Adds a new tree with a `UserNode` as the root.
    - compute_correct_trajectory(): Computes the correct trajectory from root to the last child in each tree.
    """
    init_cells: Optional[List[CellUnion]] = None

    cells: List[CellUnion] = Field(default=[])
    node_trees: List[NodeUnion] = Field(default=[])
    current_node: Optional[NodeUnion] = None

    cell2node: dict[CellUnion, NodeUnion] = Field(default={})
    id2node: dict[uuid.UUID, NodeUnion] = Field(default={})

    def initialize(self, init_cells: List[NotebookCell]):
        self.init_cells = init_cells
        self.cells.extend(init_cells)

    def add_messages(self, node: Node, action_signal: str = AWAIT_TAG) -> None:

        # self.id2node
        self.id2node[node.id] = node

        if isinstance(node, UserNode):
            if action_signal == AWAIT_TAG:
                node.children_node_ids = []
                self.node_trees.append(node)
                self.current_node = node

                if isinstance(node.cells_generated, NotebookCell):
                    node_cells = [node.cells_generated]
                else:
                    node_cells = node.cells_generated
                self.cells.extend(node_cells)

                for cell in node_cells:
                    self.cell2node[cell] = node

        elif isinstance(node, StepNode) or isinstance(node, ExecutionNode):

            assert self.current_node != None

            # node
            node.children_node_ids = []
            node.parent_node_id = self.current_node.id
            if self.current_node.children_node_ids is not None:
                self.current_node.children_node_ids.append(node.id)
            else:
                self.current_node.children_node_ids = [node.id]
            self.current_node = node

            # cells
            if isinstance(node.cells_generated, NotebookCell):
                node_cells = [node.cells_generated]
            else:
                node_cells = node.cells_generated
            self.cells.extend(node_cells)

            # cells2node
            for cell in node_cells:
                self.cell2node[cell] = node

        elif isinstance(node, DebugNode):
            assert isinstance(self.current_node, ExecutionNode)
            if self.current_node.debugging_trace_ids is None:
                self.current_node.debugging_trace_ids = []

            self.current_node.debugging_trace_ids.append(node.id)

            # cells
            if isinstance(node.cells_generated, NotebookCell):
                node_cells = [node.cells_generated]
            else:
                node_cells = node.cells_generated
            self.cells.extend(node_cells)

            # cells2node
            for cell in node_cells:
                self.cell2node[cell] = node

        elif isinstance(node, PostDebuggingNode):
            assert isinstance(self.current_node, ExecutionNode)

            self.current_node.post_debugging_result_id = node.id

            # cells
            for cell in self.current_node.cells_to_debug:
                if cell in self.cells:
                    self.cells.remove(cell)
            for debug_node_id in self.current_node.debugging_trace_ids:
                debug_node = self.id2node[debug_node_id]
                for cell in debug_node.cells_generated:
                    if cell in self.cells:
                        self.cells.remove(cell)

            if isinstance(node.cells_generated, NotebookCell):
                node_cells = [node.cells_generated]
            else:
                node_cells = node.cells_generated
            self.cells.extend(node_cells)

            # cells2node
            for cell in node_cells:
                self.cell2node[cell] = node

    def to_string(self, add_sender_prefix: bool = False) -> str:
        if add_sender_prefix:
            return "\n".join(
                [
                    (
                        f"[{message.role}]: {message.content}"
                        if message.role != ""
                        else message.content
                    )
                    for message in self.cells
                ]
            )
        else:
            return "\n".join([message.content for message in self.cells])

    def reset(self) -> None:
        self.cells = []

    def _find_last_step_idx(self, cells: List[NotebookCell]):
        last_step_index = None
        last_user_index = None

        for idx, cell in enumerate(cells):
            if isinstance(cell, StepCell):
                last_step_index = idx
            if isinstance(cell, UserCell):
                last_user_index = idx

        return last_step_index if last_step_index != None else last_user_index + 1

    def find_last_step(self) -> NotebookCell | None:
        idx = self._find_last_step_idx(self.cells)
        if idx is not None:
            return self.cells[idx]
        else:
            None

    def clear_last_step(self, action_signal: str = ITERATE_ON_LAST_STEP):

        last_step_index = self._find_last_step_idx(self.cells)
        last_step_cell = self.cells[last_step_index]

        if last_step_index is not None:
            self.cells = self.cells[:last_step_index]
            last_step_node = self.cell2node.get(last_step_cell, None)
            if last_step_node != None:
                self.current_node = self.id2node[last_step_node.parent_node_id]
                logger.info(f"Current node is switched to {self.current_node.id}")
                logger.info(f"Current node: {self.current_node.to_string()}")
            else:
                logger.warn(f"Last step node not found! Current node remains.")

    def to_messages(
        self,
        format: FormatType,
        task_type: Literal[
            "Initiate Step",
            "Incremental Execution Stage",
            "Planning Stage",
            "Debugging Stage",
            "Post-debugging Stage",
        ] = "Initiate Step",
    ) -> List[dict]:
        """
        Convert the chat history to a list of OpenAI messages.

        Parameters:
        - format: Specifies the format type for message conversion.
        - insert_before_last_step_prompt: Optional system prompt to insert before the last step.

        Returns:
        A list of dictionaries representing OpenAI messages.
        """

        if task_type in (
            "Initiate Step",
            "Incremental Execution Stage",
            "Planning Stage",
        ):

            organized_cells = self.cells
            messages = self.organize_cells_to_messages(organized_cells, format=format)

        elif task_type in ("Debugging Stage", "Post-debugging Stage"):
            organized_cells = []
            organized_cells.extend(self.cells)
            assert self.current_node != None and (
                (
                    isinstance(self.current_node, ExecutionNode)
                    or isinstance(self.current_node, StepNode)
                )
                and len(self.current_node.cells_to_debug) != 0
            )

            # sort the order of correct cells
            if len(self.current_node.correct_cells) != 0:
                insert_idx = None
                for correct_cell in self.current_node.correct_cells:
                    try:
                        current_idx = self.cells.index(correct_cell)

                    except Exception as e:
                        break

                    if insert_idx is None:
                        insert_idx = current_idx
                    else:
                        insert_idx += 1

                    if current_idx != insert_idx:
                        cell = self.cells.pop(current_idx)
                        self.cells.insert(insert_idx, cell)

            cell_to_insert_before = self.current_node.cells_to_debug[0]

            cell_to_insert_before = organized_cells.index(cell_to_insert_before)
            if task_type == "Debugging Stage":
                organized_cells.insert(
                    cell_to_insert_before,
                    MarkdownCell(
                        content=DEBUGGING_SYSTEM_PROMPT, role="system", name="System"
                    ),
                )
            elif task_type == "Post-debugging Stage":
                organized_cells.insert(
                    cell_to_insert_before,
                    MarkdownCell(
                        content=POST_DEBUGGING_SYSTEM_PROMPT,
                        role="system",
                        name="System",
                    ),
                )

            messages = self.organize_cells_to_messages(organized_cells, format=format)

        return messages

    def organize_cells_to_messages(
        self,
        organized_cells: List[NotebookCell],
        format: FormatType,
    ):

        openai_messages: list[dict] = []
        assistant_outputs: list[str] = []
        code_outputs: list[str] = []
        def merge_and_push_code_outputs():
            if code_outputs:
                merged_message = {
                    "role": "user",
                    "name": "Jupyter_Kernel",
                    "content": "\n".join(code_outputs),
                }
                openai_messages.append(merged_message)
                code_outputs.clear()

        def merge_and_push_assistant_outputs():
            if assistant_outputs:
                merged_message = {
                    "role": "assistant",
                    "name": "Datawise_Agent",
                    "content": "\n".join(assistant_outputs),
                }
                openai_messages.append(merged_message)
                assistant_outputs.clear()

        for cell in organized_cells:
            if isinstance(cell, MarkdownCell):
                if cell.role in ("system", "user"):

                    merge_and_push_assistant_outputs()
                    merge_and_push_code_outputs()

                    message = {
                        "role": cell.role,
                        "name": cell.name,
                        "content": cell.to_string(format=format),
                    }
                    openai_messages.append(message)
                elif cell.role == "assistant":
                    merge_and_push_code_outputs()

                    assistant_outputs.append(cell.to_string(format=format))

            elif isinstance(cell, CodeCell):
                merge_and_push_code_outputs()
                assistant_outputs.append(cell.to_string(format=format))

            elif isinstance(cell, CodeOutputCell):
                merge_and_push_assistant_outputs()
                code_outputs.append(cell.to_string(format=format))
        merge_and_push_assistant_outputs()
        merge_and_push_code_outputs()

        return openai_messages

    @classmethod
    def from_json(
        cls,
        obj: Any,
    ):
        cellunion_adapter = TypeAdapter(CellUnion)
        nodeunion_adapter = TypeAdapter(NodeUnion)
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

        _id2cells = {}
        # 恢复 init_cells
        init_cells = [
            cellunion_adapter.validate_python(cell)
            for cell in data.get("init_cells", [])
        ]
        for cell in init_cells:
            _id2cells[cell.id] = cell

        # 恢复 id2node
        # 恢复 cell2node
        id2node = {}
        cell2node = {}
        nodes_data = data.get("id2node", {})
        for node_id_str, node_data in nodes_data.items():
            node_id = uuid.UUID(node_id_str)
            node = nodeunion_adapter.validate_python(node_data)
            id2node[node_id] = node

            if (
                isinstance(node, UserNode)
                or isinstance(node, DebugNode)
                or isinstance(node, PostDebuggingNode)
            ):
                if not isinstance(node.cells_generated, list):
                    cells_generated = [node.cells_generated]
                else:
                    cells_generated = node.cells_generated

                for cell in cells_generated:
                    _id2cells[cell.id] = cell
                    cell2node[cell] = node

            elif isinstance(node, ExecutionNode) or isinstance(node, StepNode):
                if not isinstance(node.cells_generated, list):
                    cells_generated = [node.cells_generated]
                else:
                    cells_generated = node.cells_generated
                for cell in cells_generated:
                    _id2cells[cell.id] = cell
                    cell2node[cell] = node

                # correct_cells
                if node.correct_cells != None:
                    new_correct_cells = []
                    for cell in node.correct_cells:
                        cell_fetch = _id2cells.get(cell.id)
                        if cell_fetch:
                            # has existed
                            new_correct_cells.append(cell_fetch)
                        else:
                            new_correct_cells.append(cell)
                            # new create
                            _id2cells[cell.id] = cell
                            cell2node[cell] = node
                    node.correct_cells = new_correct_cells

                # cells_to_debug
                if node.cells_to_debug != None:
                    new_cells_to_debug = []
                    for cell in node.cells_to_debug:
                        cell_fetch = _id2cells.get(cell.id)
                        if cell_fetch:
                            # has existed
                            new_cells_to_debug.append(cell_fetch)
                        else:
                            new_cells_to_debug.append(cell)
                            # new create
                            _id2cells[cell.id] = cell
                            cell2node[cell] = node
                    node.cells_to_debug = new_cells_to_debug
        # clean all the CodeOutputCells
        for cell_id, cell in _id2cells.items():
            if isinstance(cell, CodeCell):
                code_output_cell = cell.code_output
                if code_output_cell != None:
                    code_output_cell_id = code_output_cell.id
                    if code_output_cell_id in _id2cells:
                        cell.code_output = _id2cells[code_output_cell_id]
        # 恢复current_node
        current_node_data = data.get("current_node", None)
        if not current_node_data:
            current_node = None
        else:
            node_id = uuid.UUID(current_node_data["id"])
            current_node = id2node.get(node_id)

        # 恢复 node_trees
        node_trees = []
        for node_data in data.get("node_trees", []):
            node_id = uuid.UUID(node_data["id"])
            node = id2node.get(node_id)
            # if not node:
            #    # 如果 node 不存在，则创建新的 node
            #    node = nodeunion_adapter.validate_python(node_data)
            assert (
                node is not None
            ), "New created node has not been registered! Out of Control!"
            node_trees.append(node)

        # 恢复 cells
        cells = []
        for cell_data in data.get("cells", []):
            cell = _id2cells.get(uuid.UUID(cell_data["id"]))
            if not cell:
                logger.warn(
                    "New created cells has not been registered! Out of Control!"
                )
                cell = cellunion_adapter.validate_python(cell_data)
                _id2cells[cell.id] = cell
            cells.append(cell)

        # 创建实例并赋值
        instance = cls(
            init_cells=init_cells,
            cells=cells,
            node_trees=node_trees,
            current_node=current_node,
            cell2node=cell2node,
            id2node=id2node,
        )

        return instance

    def fetch_response(self, mode: Literal["all", "last_one"] = "last_one"):
        # all_response_content = ""
        last_user_content = ""
        last_response_content = ""
        last_idx = None
        if mode == "last_one":
            for idx in range(len(self.cells) - 1, -1, -1):
                if isinstance(self.cells[idx], UserCell):
                    last_idx = idx
                    break

            if last_idx:
                for i, cell in enumerate(self.cells[last_idx:]):
                    if i == 0:
                        last_user_content = cell.to_string()
                    else:
                        last_response_content += cell.to_string() + "\n"

            return last_user_content, last_response_content
        else:
            raise ValueError("mode `all` is not developed.")
