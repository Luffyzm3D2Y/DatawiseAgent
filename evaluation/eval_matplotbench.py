"""
One script to evaluate DatawiseAgent on the MatplotBench.
Usage:
    First of all, start the server at http://localhost:8000

    Example:
        python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8" [--with_tool]

Arguments:
    --note       Unique identifier for the run (e.g., "temperature-0_2-args-7-6-8")
    --with_tool  (Optional) Enable visual tool during evaluation

"""

import json
import os
import uuid
from uuid import UUID
import argparse
from pathlib import Path
from chat_test_asyncio import create_session, create_user, chat, upload_file
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from string import Template
import glob

# w/ visual tool
USER_PROMPT_WITH_TOOL = '''You are asked to complete a task, and you also might need to process some related sheets, which will be metioned in query.If the file paths are given, you need to extract informations from these files to complete the task.
Here is the user query: [User Query]:
"""
${instruction}
"""
If the query requires data manipulation from a csv file, process the data from the csv file and draw the plot. When you complete a plot, remember to save it to a png file.

You can use the `evaluate_image` AI visual tool to obtain visual feedback on the image only as a reference. Continue this process of refinement until the visualization perfectly fulfills all criteria.
'''

# w/o visual tool
USER_PROMPT_WITHOUT_TOOL = '''You are asked to complete a task, and you also might need to process some related sheets, which will be metioned in query.If the file paths are given, you need to extract informations from these files to complete the task.
Here is the user query: [User Query]:
"""
${instruction}
"""
If the query requires data manipulation from a csv file, process the data from the csv file and draw the plot. When you complete a plot, remember to save it to a png file.
'''

from pathlib import Path
from PIL import Image
import numpy as np


def is_blank_image(path, tolerance=0):
    """
    判断图片是否为空白图片。

    参数:
    - path (str): 图片文件路径。
    - tolerance (int): 颜色容差，默认为0表示完全相同。

    返回:
    - bool: 如果图片为空白，返回True；否则返回False。
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")  # 转换为RGB模式
            np_img = np.array(img)
            first_pixel = np_img[0, 0]
            if tolerance == 0:
                return np.all(np_img == first_pixel)
            else:
                diff = np.abs(np_img - first_pixel)
                return np.all(diff <= tolerance)
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return False


def get_latest_non_blank_image(directory, tolerance=0):
    """
    获取指定目录中最新的非空白图片。

    参数:
    - directory (str): 目标目录路径。
    - tolerance (int): 颜色容差，默认为0。

    返回:
    - str or None: 最新的非空白图片路径，如果不存在则返回None。
    """
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return None

    # 支持的图片扩展名
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"]

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    if not image_files:
        print(f"No images found in directory: {directory}")
        return None

    # 按修改时间降序排序
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # 查找第一个非空白图片
    for image_path in image_files:
        if not is_blank_image(image_path, tolerance):
            return image_path

    # 如果所有图片都是空白的
    print(f"All images in {directory} are blank.")
    return None


def determine_result_path(user_id, session_id, ground_truth_path, tolerance=0):
    """
    确定result_path的函数。

    参数:
    - user_id (str or int): 用户ID。
    - session_id (str or int): 会话ID。
    - ground_truth_path (str): 默认的ground truth路径。
    - tolerance (int): 颜色容差，默认为0。

    返回:
    - str: 确定的result_path。
    """
    base_path = os.path.join("../log/workspace/users", str(user_id), str(session_id))

    working_dir = os.path.join(base_path, "working")
    display_dir = os.path.join(base_path, "display")

    # 尝试从working目录获取最新的非空白图片
    result_path = get_latest_non_blank_image(working_dir, tolerance)
    if result_path:
        return result_path

    # 如果working目录没有，尝试从base目录获取
    result_path = get_latest_non_blank_image(base_path, tolerance)
    if result_path:
        return result_path

    # 如果base目录没有，尝试从display目录获取
    result_path = get_latest_non_blank_image(display_dir, tolerance)
    if result_path:
        return result_path

    # 如果两者都没有，使用默认路径
    default_path = os.path.join(ground_truth_path, "empty.png")
    if os.path.exists(default_path):
        return default_path
    else:
        print(f"Default image not found at {default_path}.")
        return None


def main(user_id: UUID, with_tool: bool = False):

    if with_tool:
        user_prompt_template = USER_PROMPT_WITH_TOOL
    else:
        user_prompt_template = USER_PROMPT_WITHOUT_TOOL

    data_path = "./MatplotBench/data"
    result_file_path = "experimental_results/MatplotBench/gpt-4o-mini/result.jsonl"

    log_file_path = "experimental_results/MatplotBench/gpt-4o-mini/process.log"

    ground_truth_path = "./MatplotBench/data/ground_truth"
    # open the json file
    data = json.load(open(f"{data_path}/benchmark_instructions.json"))

    result_dict = {}

    if not os.path.exists(result_file_path):
        Path(result_file_path).touch()

    with open(result_file_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)

            id = item["id"]
            result_dict[id] = item
    print(len(result_dict))

    for i, item in enumerate(tqdm(data)):
        novice_instruction = item["simple_instruction"]
        expert_instruction = item["expert_instruction"]
        example_id = item["id"]

        example_directory = f"{data_path}/data/{example_id}"

        if example_id in result_dict:
            continue

        try:
            # create session
            session_id = UUID(
                create_session(
                    user_id, session_name=str(example_id), tool_mode="dsbench"
                )
            )
            print(f"create session: {session_id} for example {example_id}")

            # upload_file(user_id, session_id, )
            # upload_file()
            for root, dirs, files in os.walk(example_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, example_directory)
                    upload_file(str(user_id), str(session_id), file_path, rel_path)
            query = Template(user_prompt_template).safe_substitute(
                {"instruction": novice_instruction}
            )
            chat_response = chat(str(user_id), str(session_id), query=query)

            user_content = chat_response["user_content"]
            response_content = chat_response["response_content"]

            result_path = determine_result_path(
                user_id,
                session_id,
                ground_truth_path,
            )

            iteration_result = {
                "id": example_id,
                "input_text": query,
                "instruction": novice_instruction,
                "user_id": str(user_id),
                "session_id": str(session_id),
                "response": response_content,
                "ground_truth_path": os.path.join(
                    ground_truth_path, f"example_{example_id}.png"
                ),
                "result_path": result_path,
            }

            with open(result_file_path, encoding="utf-8", mode="a+") as f:
                f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                f.flush()

        except Exception as e:
            import traceback

            error_msg = traceback.format_exc()
            with open(log_file_path, "a") as log_file:
                log_file.write(f"ERROR: {error_msg}\n")
                log_file.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run datawiseagent evaluation.")

    parser.add_argument(
        "--note",
        type=str,
        required=True,
        help="Readable note/username identifier (e.g., MatplotBench-temperature-0-args-(7,6,8)-for-loop-gpt-4o-mini)",
    )
    parser.add_argument(
        "--with_tool",
        action="store_true",
        help="Enable visual tool (default: disabled)",
    )

    args = parser.parse_args()

    user_id = create_user(username=args.note)
    print(f"Created user_id: {user_id}")

    main(user_id, with_tool=args.with_tool)
