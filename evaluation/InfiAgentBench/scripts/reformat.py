"""
Script: reformat.py
-------------------
Working directory: `./evaluation` (relative to the root directory of the project).

Usage:
    ```bash
    python ./InfiAgentBench/scripts/reformat.py \
        --model <MODEL_NAME> \
        --responses_file_path ./data/<FILE_PATH> \
        --output_file_path ./InfiAgentBench/reformat/gpt_4o_mini/<FILE_PATH>
    ```

Example:
    ```bash
    python ./InfiAgentBench/scripts/reformat.py \
        --model "gpt-4o-mini" \
        --responses_file_path ./data/results_datawise-test.jsonl \
        --output_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl
    ```
"""

import logging
import time
import json
import argparse
import traceback
import os

import requests

from utils.utils import read_jsonl, write_jsonl
from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL")
)
def define_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url_file", type=str, default="./InfiAgentBench/scripts/url.txt"
    )
    parser.add_argument(
        "--api_key_file", type=str, default="./InfiAgentBench/scripts/api_key.txt"
    )
    parser.add_argument(
        "--questions_file_path",
        type=str,
        default="./InfiAgentBench/data/da-dev-questions.jsonl",
    )
    parser.add_argument(
        "--responses_file_path",
        type=str,
        default="./data/results_datawise_gpt-4o-mini-2024-07-18.jsonl",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_resp", type=int, default=2048)

    args = parser.parse_args()
    return args


def call(messages, args):
    data = {
        "max_tokens": args.max_resp,
        "model": args.model,
        "temperature": 0,
        "messages": messages,
    }
    while True:
        try:
            # response = requests.post(args.url, headers=headers, data=json.dumps(data))
            # import pdb

            # pdb.set_trace()
            # result = response.json()
            print(args.model)
            result = client.chat.completions.create(**data)

            return result.choices[0].message.content
        except Exception as e:
            logging.error(data)
            logging.error(traceback.format_exc())
            time.sleep(10)


demons = """\Format{{
@shapiro_wilk_statistic[test_statistic]
@shapiro_wilk_p_value[p_value]
where "test_statistic" is a number between 0 and 1 representing the Shapiro-Wilk test statistic. Rounding off the answer to two decimal places.
where "p_value" is a number between 0 and 1 representing the p-value from the Shapiro-Wilk test. Rounding off the answer to four decimal places.
}}
\Answer{{
@shapiro_wilk_statistic[0.56]
@shapiro_wilk_p_value[0.0002]   
}}

\Format{{
@total_votes_outliers_num[outlier_num]
where "outlier_num" is an integer representing the number of values considered outliers in the 'total_votes' column.
}}
\Answer{{
@total_votes_outliers[10]   
}}
"""

reformat_template = """You should strictly follow the output requirements in the Format part. Here're some examples: 
{demons}. 
Your answer should contain all the \"@answer_name[answer]\" in the order mentioned, each \"answer\" should be in the range of value as required. 
The format requirements of this question is:
{format}. Please give your answer:"""

if __name__ == "__main__":
    args = define_arguments()
    args.url = open(args.url_file).read().strip()
    args.api_key = open(args.api_key_file).read().strip()

    questions = read_jsonl(args.questions_file_path)

    responses = []
    with open(args.responses_file_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            responses.append(item)

    reformatted_responses_ids = []
    from pathlib import Path
    import os

    output_file_path = Path(args.output_file_path)
    if not os.path.exists(output_file_path):
        try:
            # 创建空文件
            output_file_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            output_file_path.touch()  # 创建空文件
            # 创建文件并写入空字典
            # with output_file_path.open("w", encoding="utf-8") as f:
            #    json.dump({}, f, indent=4)
            print(f"Created empty file at: {output_file_path}")
        except Exception as e:
            print(f"Failed to create file: {e}")
    else:
        print(f"File already exists: {output_file_path}")

    with open(args.output_file_path, encoding="utf-8", mode="r") as f:
        # reformatted_responses = json.load(f)
        for line in f:
            item = json.loads(line)
            if "id" in item:
                reformatted_responses_ids.append(item["id"])

    # response_number = len(reformatted_responses_ids)
    new_responses = []

    for res_id, response in enumerate(responses):
        if response["id"] in reformatted_responses_ids:
            continue

        for question in questions:
            if question["id"] == response["id"]:
                question_description = question["question"]
                format = question["format"]
                break

        messages = [{"role": "user", "content": question_description}]
        messages.append({"role": "assistant", "content": response["response"]})
        messages.append(
            {
                "role": "user",
                "content": reformat_template.format(demons=demons, format=format),
            }
        )

        try:
            if "is_solved" in response and response["is_solved"]:
                pass

            else:
                reformatted_response = call(messages, args)
                response["reformat_response"] = reformatted_response

                print(f"{res_id}: {reformatted_response}")

            new_responses.append(response)
            write_jsonl([response], args.output_file_path, mode="a+")
        except Exception as e:
            import traceback

            traceback.print_exc()
            # import pdb

            # pdb.set_trace()
