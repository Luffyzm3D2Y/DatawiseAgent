"""
score4each.py
=============

This script computes evaluation scores for each task in the DataModeling benchmark.

It loads the ground truth answers, model prediction results, and runs the
corresponding evaluation script for each dataset.

Usage
-----
First, ensure that model outputs have been generated and saved in:
    ./experimental_results/Datamodeling-DSBench/<MODEL>/results.jsonl

Then run this script with the desired model name:

    python ./DataModeling/scripts/score4each.py --model gpt-4o-mini
    python ./DataModeling/scripts/score4each.py --model qwen25-72B

Arguments
---------
--model : str
    Model name (e.g., gpt-4o-mini, qwen25-72B).
    Used to locate the results.jsonl file and store evaluation outputs.

Outputs
-------
- Evaluation results will be written under:
    ./results/DataModeling/<MODEL>/performances/<TASK_NAME>/
- Each evaluation script (`*_eval.py`) will be called automatically.
"""

import os
import json
from tqdm import tqdm
import time
import argparse
from pathlib import Path


data = []
with open("./DataModeling/data.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(len(data))

parser = argparse.ArgumentParser(description="Run DataModeling evaluation.")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model name (e.g., gpt-4o-mini, qwen25-72B)",
)
args = parser.parse_args()

model = args.model

# assume the final submission file will be definitely named as `submission.csv`
gt_path = "./DataModeling/data/answers/"
output_dir = f"./experimental_results/Datamodeling-DSBench/{model}/performances/"

results_path = f"./experimental_results/Datamodeling-DSBench/{model}/results.jsonl"
python_path = "./DataModeling/evaluation/"

print(f"Using model: {model}")
print(f"Ground truth path: {gt_path}")
print(f"Output dir: {output_dir}")
print(f"Results path: {results_path}")
print(f"Evaluation script path: {python_path}")


results = []
with open(results_path, encoding="utf-8", mode="r") as f:
    for line in f:
        results.append(json.loads(line))

print(f"existing result number: {len(results)}")

for result in results:
    pred_file = result["result_path"]
    name = result["name"]
    answer_file = gt_path + name + "/test_answer.csv"
    try:
        if os.path.exists(os.path.join(output_dir, name)):
            continue

        if not os.path.exists(os.path.join(output_dir, name)):
            os.makedirs(os.path.join(output_dir, name))

        print(f"compute performance for {name}")
        os.system(
            f"python {python_path}{name}_eval.py --answer_file {answer_file} --predict_file {pred_file} --path {output_dir} --name {name}"
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
