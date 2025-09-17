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
import glob
import httpx

def main(
    user_id: uuid.UUID,
    user_name: Optional[str] = None,
    result_path: str | Path = Path(f"./results/DataModeling/gpt-4o-mini/"),
    model: str = "gpt-4o-mini",
):

    Path(result_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(result_path) / "results.jsonl"

    log_path = Path(result_path) / "log.txt"

    if not os.path.exists(results_path):
        Path(results_path).touch()

    if not os.path.exists(log_path):
        Path(log_path).touch()

    id2results = {}
    import pdb

    pdb.set_trace()
    with open(results_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            id2results[item["id"]] = item

    samples = []
    with open("./DataModeling/data.jsonl", encoding="utf-8", mode="r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"samples count: {len(samples)}")

    print(f"id2results count: {len(id2results)}")

    import pdb

    pdb.set_trace()

    instruction = "I have a data modeling task. You must give me the predicted results as a CSV file as detailed in the following content. You should try your best to predict the answer. I provide you with three files. One is training data, one is test data. There is also a sample file for submission."

    for i, item in enumerate(samples):

        if i in id2results:
            continue

        if i == 3:  #  or i == 17:
            continue

        # if i == 14 or i == 17 or i == 22 or i == 33 or i == 48:
        #     print(item)
        #     continue

        name = item["name"]
        with open(f"./DataModeling/data/task/{name}.txt", "r") as f:
            description = f.read()
        text = f"All three data files can be found in the folder `./input/`. After the data modeling, please give me the prediction results for the test file. You must run the code for the predicted results and save the answer as a csv file following the format of the given sample file for submission. The final submission file should be saved in the path `./working/submission.csv`."

        text = f"""All three data files can be found in the folder `./input/`. **You should use sklearn or pytorch to complete the task.** Any training scripts, models, and experiment log should be saved in `./input/`. After data modeling, provide the prediction results for the test file in the format specified by the sample submission file. Save the final submission to `./input/final_submission.csv`.
        """

        all_context = instruction + "\n" + description + "\n" + text

        try:
            session_id = uuid.UUID(
                create_session(
                    user_id=user_id, session_name=name, tool_mode="datamodeling"
                )
            )
            print(f"create session:{session_id} for case {name} with id {name}")

            data_split_path = f"./DataModeling/data/data_resplit/{name}"
            for root, dirs, files in os.walk(data_split_path):
                for file in files:
                    if file.endswith(".csv"):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, data_split_path)

                        upload_file(
                            str(user_id),
                            str(session_id),
                            file_path=full_path,
                            filename_to_save=relative_path,
                        )

            start = time.time()

            chat_response = None
            time_consumed = None

            try:
                chat_response = chat(
                    str(user_id),
                    str(session_id),
                    query=all_context,
                    work_mode="jupyter+script",
                )
            except httpx.TimeoutException:
                print("chat() 调用超时，自动设置 chat_response")
                chat_response = {
                    "user_content": all_context,
                    "response_content": "TIMEOUT=3600",
                }
                time_consumed = 3600

            end = time.time()

            if not time_consumed:
                time_consumed = end - start

            user_content = chat_response["user_content"]
            response_content = chat_response["response_content"]

            input_dir = f"../log/workspace/users/{user_id}/{session_id}/input/"
            work_dir = f"../log/workspace/users/{user_id}/{session_id}/working/"

            submission_files = []
            submission_files.extend(
                glob.glob(os.path.join(input_dir, "*submission*.csv"))
            )
            submission_files.extend(
                glob.glob(os.path.join(work_dir, "*submission*.csv"))
            )

            if submission_files:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

                if not os.path.exists(submission_file_path):
                    submission_files.sort(
                        key=lambda x: os.path.getmtime(x), reverse=True
                    )
                    submission_file_path = submission_files[0]

            else:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

            iteration_result = {
                "id": i,
                "name": name,
                "input_text": all_context,
                "response": response_content,
                "session_id": str(session_id),
                "user_id": str(user_id),
                "time": time_consumed,
                "cost": None,
                "model": model,
                "input": None,
                "output": None,
                "result_path": submission_file_path,
            }
            print(iteration_result)
            print(f"submission_file_path:{submission_file_path}")

            with open(results_path, encoding="utf-8", mode="a+") as f:
                f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                f.flush()

        except Exception as e:
            import traceback

            with open(log_path, encoding="utf-8", mode="a+") as f:
                f.write(traceback.format_exc())

        # create session

        # upload files

        start = time.time()
        cost = 0
        error = ""

    pass


if __name__ == "__main__":
    import uuid

    # user_name = "DataModeling-qwen2.5-temperature=0-args=(7,6,8)-for-loop"
    user_name = "test_debug"
    user_id = create_user(username=user_name)
    # user_id = uuid.UUID("383bf999-5537-4267-a65a-a84012b32101")

    print(user_id)
    main(
        user_id=user_id,
        user_name=user_name,
        result_path="./results/DataModeling/gpt-4o-test/",
        # result_path="./results/DataModeling/qwen25-72B/",
        # model="Qwen/Qwen2.5-72B-Instruct",
    )
