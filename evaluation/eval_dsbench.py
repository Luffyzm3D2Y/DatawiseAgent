"""
Run DSBench Evaluation
======================

This script evaluates models on DSBench by creating a user, reading dataset samples,
uploading related files, and generating responses with chat sessions.

Usage
-----
First, ensure the server is running (default: http://localhost:8000).

Basic usage with defaults:
    python eval_dsbench.py

Specify custom parameters:
    python eval_dsbench.py \
        --user_name "DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop" \
        --result_path "./results/DSBench/gpt-4o-mini/"

Arguments
---------
--user_name : str
    Name of the user as readable identifier(default: DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)

--result_path : str
    Directory where results will be stored (default: ./results/DSBench/gpt-4o-mini/)
"""

import json
import os
import uuid
import argparse
from pathlib import Path
from chat_test_asyncio import create_session, create_user, chat, upload_file
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent
from IPython.display import Image, display

# import fitz  # PyMuPDF
import json
import base64
import re
import time
import pandas as pd
from tqdm import tqdm
from typing import Optional


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def find_jpg_files(directory):
    jpg_files = [
        file
        for file in os.listdir(directory)
        if file.lower().endswith(".jpg") or file.lower().endswith(".png")
    ]
    return jpg_files if jpg_files else None


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def find_excel_files(directory):
    jpg_files = [
        file
        for file in os.listdir(directory)
        if (
            file.lower().endswith("xlsx")
            or file.lower().endswith("xlsb")
            or file.lower().endswith("xlsm")
        )
        and not "answer" in file.lower()
    ]
    return jpg_files if jpg_files else None


def read_excel(file_path):
    # 读取Excel文件中的所有sheet
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets


def dataframe_to_text(df):
    # 将DataFrame转换为文本
    text = df.to_string(index=False)
    return text


def combine_sheets_text(sheets):
    # 将所有sheet的文本内容组合起来
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text


def read_txt(path):
    with open(path, "r") as f:
        return f.read()


def truncate_text(text, max_tokens=128000):
    # 计算当前文本的token数
    tokens = text.split()
    if len(tokens) > max_tokens:
        # 截断文本以确保不超过最大token数
        text = " ".join(tokens[-max_tokens:])
    return text


def concurrent_main(user_id: uuid.UUID, num_workers: int = 4):
    pass


def read_jsonl_files(result_path: str | Path):
    # 确保路径是 Path 类型
    result_path = Path(result_path)

    # 用于存储最终的字典，键是(id, question_id)，值是iteration_result
    result_dict = {}

    # 遍历目录下的所有jsonl文件
    for jsonl_file in result_path.glob("*.jsonl"):
        # 打开每个jsonl文件
        with open(jsonl_file, "r") as file:
            # 遍历文件中的每一行
            for line in file:
                try:
                    # 解析每一行的json数据
                    iteration_result = json.loads(line.strip())

                    # 提取所需的字段
                    key = (iteration_result["id"], iteration_result["question_id"])

                    # 将iteration_result添加到字典中
                    result_dict[key] = iteration_result

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {jsonl_file}: {e}")
                except KeyError as e:
                    print(f"Missing expected key {e} in file {jsonl_file}")

    return result_dict


def main(
    user_id: uuid.UUID,
    user_name: Optional[str] = None,
    result_path: str | Path = Path(f"./results/DSBench/gpt-4o-mini/"),
):
    Path(result_path).mkdir(parents=True, exist_ok=True)

    result_dict = read_jsonl_files(result_path)

    samples = []
    data_path = "./DSBench/data"
    with open("./DSBench/data.jsonl", "r") as f:
        for line in f:
            samples.append(eval(line.strip()))
    len(samples)

    for sample in samples:
        image = find_jpg_files(os.path.join(data_path, sample["id"]))

        excels = find_excel_files(os.path.join(data_path, sample["id"]))

        introduction = read_txt(
            os.path.join(data_path, sample["id"], "introduction.txt")
        )
        questions = []
        for question_name in sample["questions"]:
            questions.append(
                (
                    question_name,
                    read_txt(
                        os.path.join(data_path, sample["id"], question_name + ".txt")
                    ),
                )
            )

        text = f"The introduction is detailed as follows. \n {introduction}"
        if excels:
            text += "\n\nI have uploaded the Excel files. You should collect information from excel files. And if it's too complex and unstructured to extract data by writing code, you could just read the pure format content (like by dataframe.to_string(index=False)) to get an easy content review. The worksheets can be found in the ./input directory: "
            for excel in excels:
                text += f" {excel}"
        if image:
            text += f"\nI have uploaded the image files to help you understand the context and question better, and you could use the function `ask_about_images` to understand the images. The image can be obtained in the `./input` directory: {image[0]} \n"

        # print(workbooks)
        answers = []
        for question_name, question in tqdm(questions):
            # question_content += question

            # sample['id']
            # question_name
            print(f"currently processing { str( (sample['id'], question_name) )}")
            if (sample["id"], question_name) in result_dict:
                continue

            all_context = (
                text
                + f"\n\n\n\n{question}\nNote: Please read the materials provided and answer the given question. **Once you get the answer from the question or you consider it's too hard to complete the task after multiple explorations, you should terminate.**"
            )
            input_t = all_context
            # input_t = truncate_text(all_context, 2000)

            try:
                case_name = str((sample["id"], question_name))
                session_id = uuid.UUID(
                    create_session(
                        user_id=user_id, session_name=case_name, tool_mode="dsbench"
                    )
                )
                print(f"create session:{session_id} for case {case_name}")

                case_path = Path(os.path.join(data_path, sample["id"]))
                questions_filenames = [item[0] + ".txt" for item in questions]
                # upload files
                for root, dirs, files in os.walk(case_path):
                    for file in files:
                        if (
                            file.endswith("introduction.txt")
                            or file in questions_filenames
                            or file.startswith(".")
                        ):
                            continue
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, case_path)

                        upload_file(
                            str(user_id),
                            str(session_id),
                            file_path=full_path,
                            filename_to_save=relative_path,
                        )
                start = time.time()
                chat_response = chat(str(user_id), str(session_id), query=input_t)
                end = time.time()

                user_content = chat_response["user_content"]
                response_content = chat_response["response_content"]
                iteration_result = {
                    "id": sample["id"],
                    "competition_name": sample["name"],
                    "question_id": question_name,
                    "user_name": user_name,
                    "input_text": input_t,
                    "response": response_content,
                    "session_id": str(session_id),
                    "time": end - start,
                    "cost": None,
                    "model": "gpt-4o-mini",
                    "input": None,
                    "output": None,
                }

                # create new session
                # upload files
                # chat
                # get the response

                result_json_path = result_path / f"{sample['id']}.jsonl"
                with open(result_json_path, "a+") as f:
                    f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                import traceback

                traceback.print_exc()

                pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DSBench evaluation.")

    parser.add_argument(
        "--user_name",
        type=str,
        default="DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop",
        help="Name of the user (default: DSBench-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)",
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default="./results/DSBench/gpt-4o-mini/",
        help="Path to store results (default: ./results/DSBench/gpt-4o-mini/)",
    )

    args = parser.parse_args()
    user_id = create_user(username=args.user_name)
    print(f"Created user_id: {user_id}")
    main(user_id=user_id, user_name=args.user_name, result_path=args.result_path)
