import re
from typing import List, Tuple, Union

CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
UNKNOWN = "unknown"


def extract_code(
    text: Union[str, List],
    pattern: str = CODE_BLOCK_PATTERN,
    detect_single_line_code: bool = False,
) -> List[Tuple[str, str]]:
    """Extract code from a text.

    Args:
        text (str or List): The content to extract code from. The content can be
            a string or a list, as returned by standard GPT or multimodal GPT.
        pattern (str, optional): The regular expression pattern for finding the
            code block. Defaults to CODE_BLOCK_PATTERN.
        detect_single_line_code (bool, optional): Enable the new feature for
            extracting single line code. Defaults to False.

    Returns:
        list: A list of tuples, each containing the language and the code.
          If there is no code block in the input text, the language would be "unknown".
          If there is code block but the language is not specified, the language would be "".
    """
    if not detect_single_line_code:
        match = re.findall(pattern, text, flags=re.DOTALL)
        return match if match else [(UNKNOWN, text)]

    # Extract both multi-line and single-line code block, separated by the | operator
    # `([^`]+)`: Matches inline code.
    code_pattern = re.compile(CODE_BLOCK_PATTERN + r"|`([^`]+)`")
    code_blocks = code_pattern.findall(text)

    # Extract the individual code blocks and languages from the matched groups
    extracted = []
    for lang, group1, group2 in code_blocks:
        if group1:
            extracted.append((lang.strip(), group1.strip()))
        elif group2:
            extracted.append(("", group2.strip()))

    return extracted
