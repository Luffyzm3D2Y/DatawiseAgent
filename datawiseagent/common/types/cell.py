from __future__ import annotations
from typing import Any, Optional, Literal, List, Tuple, Union, Annotated
from pydantic import BaseModel, model_validator, Field, ConfigDict
from abc import ABC, abstractmethod
import regex as re
from enum import Enum
import uuid

from datawiseagent.common.types import LLMResult
from datawiseagent.coding.code_utils import (
    content_str,
    CODE_BLOCK_PATTERN,
    extract_code,
)
from datawiseagent.coding import CodeResult, CodeBlock
from datawiseagent.prompts.datawise import STEP_GOAL, USER_NOUN_TAG

class FormatType(Enum):
    PRESENT_CELLS = "present cells"
    PRESENT_PLAIN_TEXT = "display_text"


class ConvertType(Enum):
    CONVERT_CELLS = "convert_cell"
    CONVERT_USER_CELL = "convert_user_cell"


def extract_code_and_text_blocks(
    content: str, pattern: str = CODE_BLOCK_PATTERN
) -> List[Tuple[str, str]]:
    """
    Extracts both code blocks (code cells) and text (markdown cells) from content.

    Args:
        content (str): The content to extract from, expected to be a single string.
        pattern (str, optional): The regex pattern for finding code blocks.

    Returns:
        List[Tuple[str, str]]: A list of tuples where the first element is the cell type
                               ("code" or "markdown") and the second element is the content.
    """
    results = []
    last_end = 0  # Track the end of the last match

    # Find all code blocks with the specified pattern
    for match in re.finditer(pattern, content, flags=re.DOTALL):
        lang, code = match.groups()  # Extract language and code content
        # Specify lang as `python` in default for now.
        # TODO: what if `lang` is not specified?
        code = code.strip()

        # Get text (markdown) before this code block
        if match.start() > last_end:
            markdown_text = content[last_end : match.start()].strip()
            if markdown_text:
                results.append(("markdown", markdown_text))

        # Add the code block
        results.append(("code", code))  # Add language if detected
        last_end = match.end()

    # Any remaining text after the last code block is considered markdown
    if last_end < len(content):
        remaining_text = content[last_end:].strip()
        if remaining_text:
            results.append(("markdown", remaining_text))

    return results


class NotebookCell(BaseModel, ABC):
    cell_type: Literal[
        "markdown", "user_markdown", "step_markdown", "code", "codeoutput"
    ]
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str = ""
    role: Literal["user", "system", "tool", "assistant"]
    name: Optional[Literal["Datawise_Agent", "USER", "Jupyter_Kernel", "System"]]

    @classmethod
    def llm_result_convert(
        cls, llm_result: LLMResult, parse_mode: ConvertType = ConvertType.CONVERT_CELLS
    ) -> NotebookCell | List[NotebookCell]:

        if parse_mode == ConvertType.CONVERT_CELLS:
            """
            The markdown, code, code output cells are in the format below:
            ```markdown

            ```


            ```python

            ```

            Code Output:


            """
            assert llm_result.role == "assistant"

            cells = []
            results = extract_code(llm_result.content)
            for cell_type, cell_content in results:
                if cell_type == "markdown":
                    if cell_content.startswith(STEP_GOAL):
                        cells.append(
                            StepCell(
                                content=cell_content,
                                role=llm_result.role,
                                name="Datawise_Agent",
                            )
                        )
                    else:
                        cells.append(
                            MarkdownCell(
                                content=cell_content,
                                role=llm_result.role,
                                name="Datawise_Agent",
                            )
                        )

                elif cell_type == "python":
                    cells.append(
                        CodeCell(
                            content=cell_content,
                            role=llm_result.role,
                            name="Datawise_Agent",
                        )
                    )
                elif cell_type == "output":
                    pass
            return cells
        elif parse_mode == ConvertType.CONVERT_USER_CELL:

            from datawiseagent.prompts.datawise import USER_TAG

            return UserCell(
                role=llm_result.role,
                name=llm_result.name,
                content=USER_TAG + content_str(llm_result.content),
            )

    @abstractmethod
    def to_string(self, format: FormatType = FormatType.PRESENT_CELLS) -> str:
        pass

    def __hash__(self) -> int:
        return hash((self.cell_type, self.content, self.role, self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NotebookCell):
            return False
        result = True

        result &= self.content == other.content
        result &= self.cell_type == other.cell_type
        result &= self.role == other.role
        result &= self.name == other.name

        return result


class MarkdownCell(NotebookCell):
    cell_type: Literal["markdown"] = "markdown"

    # @classmethod
    # def llm_result_convert(cls, llm_result: LLMResult):
    #    return cls(content=str(llm_result.content))
    #    pass

    def to_string(self, format: FormatType = FormatType.PRESENT_CELLS) -> str:
        if format == FormatType.PRESENT_CELLS:
            if self.role == "system":
                return self.content
            else:
                return f"```markdown\n{self.content}\n```"
        else:
            return self.content


class UserCell(MarkdownCell):
    cell_type: Literal["user_markdown"] = "user_markdown"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保 StepCell 仅在内容包含 [STEP] 时有效
        if not self.content.startswith(USER_NOUN_TAG):
            raise ValueError(f"UserCell content must start with {USER_NOUN_TAG}.")


class StepCell(MarkdownCell):
    cell_type: Literal["step_markdown"] = "step_markdown"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保 StepCell 仅在内容包含 [STEP] 时有效
        if not self.content.startswith(STEP_GOAL):
            raise ValueError(f"StepCell content must start with {STEP_GOAL}.")


class CodeOutputCell(NotebookCell):
    cell_type: Literal["codeoutput"] = "codeoutput"
    code_result: Optional[CodeResult]
    role: Literal["user", "system", "tool", "assistant"] = "user"
    name: Optional[Literal["Datawise_Agent", "USER", "Jupyter_Kernel", "System"]] = (
        "Jupyter_Kernel"
    )

    # 使用 Pydantic v2 的 ConfigDict
    model_config = ConfigDict(validate_assignment=True)  # 启用赋值验证

    @model_validator(mode="after")
    def update_content(self):
        """
        在模型初始化和字段赋值后，自动更新 content 字段。
        """
        code_result = self.code_result
        if code_result is not None:
            # 使用 object.__setattr__ 直接设置属性，避免递归
            object.__setattr__(self, "content", str(code_result))
        else:
            object.__setattr__(self, "content", "")
        return self

    def to_string(self, format: FormatType = FormatType.PRESENT_CELLS) -> str:
        if format == FormatType.PRESENT_CELLS:
            return self.content
        else:
            return self.content

    def update_code_result(self, code_result: CodeResult | None):
        """
        更新 code_result，并自动更新 content。
        """
        self.code_result = code_result

    def __hash__(self) -> int:
        return hash(
            (
                self.cell_type,
                self.content,
                self.role,
                self.name,
                (self.code_result.exit_code, self.code_result.output),
            )
        )

    def __eq__(self, value: object) -> bool:
        result = super().__eq__(value)
        if result:
            if isinstance(value, CodeOutputCell):
                result &= str(self.code_result) == str(value.code_result)
                return result
            else:
                return False
        else:
            return False


class CodeCell(NotebookCell):
    cell_type: Literal["code"] = "code"
    code_output: Optional[CodeOutputCell] = None
    # assume that each one code cell is coresponding to one code output cell.

    def to_string(self, format: FormatType = FormatType.PRESENT_CELLS) -> str:
        if format == FormatType.PRESENT_CELLS:
            return f"```python\n{self.content}\n```"
        else:
            return self.content

    def to_code_block(self, lang: Literal["python"] = "python") -> CodeBlock:
        return CodeBlock(code=self.content, language=lang)

    def __hash__(self) -> int:
        return hash(
            (
                self.cell_type,
                self.content,
                self.role,
                self.name,
                self.code_output,
            )
        )

    def __eq__(self, value: object) -> bool:
        result = super().__eq__(value)
        if result and isinstance(value, CodeCell):
            result &= self.code_output == value.code_output
            return result
        return False


CellUnion = Annotated[
    Union[MarkdownCell, UserCell, StepCell, CodeOutputCell, CodeCell],
    Field(discriminator="cell_type"),
]


if __name__ == "__main__":
    result_a = CodeResult(exit_code=0, output="a")
    result_b = CodeResult(exit_code=0, output="b")
    cell = CodeOutputCell(code_result=result_a)
    import pdb

    pdb.set_trace()
    cell.code_result = result_b
    print(cell)
