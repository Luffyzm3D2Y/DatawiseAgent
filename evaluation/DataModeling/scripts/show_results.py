"""
show_results.py
===============

This script aggregates and summarizes evaluation results for the DataModeling benchmark.

It compares model predictions with ground truth (GT) and baseline performances,
computes normalized scores, and reports overall metrics such as task completion
rate, average runtime, and average performance.

Usage
-----
First, ensure evaluation results exist under:
    ./experimental_results/Datamodeling-DSBench/<MODEL>/results.jsonl
    ./experimental_results/Datamodeling-DSBench/<MODEL>/performances/<TASK_NAME>/result.txt

Then run the script with the desired model:

    python ./DataModeling/scripts/show_results.py --model gpt-4o-mini
    python ./DataModeling/scripts/show_results.py --model qwen25-72B

Arguments
---------
--model : str
    Model name (e.g., gpt-4o-mini, qwen25-72B).
    Used to locate results and evaluation outputs.

Outputs
-------
- Task completion rate (ratio of successfully evaluated tasks).
- Average time consumption across tasks.
- Average normalized performance score compared to baseline and ground truth.
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
pre_dir = f"./experimental_results/Datamodeling-DSBench/{model}/performances/"
baseline_path = "./DataModeling/save_performance/baseline"
gt_path = "./DataModeling/save_performance/GT"
results_path = f"./experimental_results/Datamodeling-DSBench/{model}/results.jsonl"


print(f"Using model: {model}")
print(f"Ground truth path: {gt_path}")
print(f"Output dir: {pre_dir}")
print(f"Results path: {results_path}")

results = []
with open(results_path, encoding="utf-8", mode="r") as f:
    for line in f:
        results.append(json.loads(line))


task_complete = 0

scores = []
all_costs = []
all_times = []


for item in results:
    try:

        flag = False  ## whetehr bigger is better
        name = item["name"]

        with open(os.path.join(gt_path, name, "result.txt"), mode="r") as f:
            gt = eval(f.read().strip())
        with open(os.path.join(baseline_path, name, "result.txt"), mode="r") as f:
            bl = eval(f.read().strip())

        # all_costs.append()
        all_times.append(item["time"])

        if gt > bl:
            flag = True

        if not os.path.exists(os.path.join(pre_dir, name, "result.txt")):
            scores.append(0)
            show_pre = "not exists"
        else:

            with open(os.path.join(pre_dir, name, "result.txt"), mode="r") as f:
                pre = eval(f.read().strip())

            if pre == "nan":
                show_pre = "nan"
                scores.append(0)
            else:
                # pre = eval(pre)
                sc = max(0, (pre - bl) / (gt - bl))
                scores.append(sc)
                show_pre = pre

            task_complete += 1

    except Exception as e:
        import traceback

        traceback.print_exc()
        import pdb

        pdb.set_trace()


print(
    f"Task completion rate is {task_complete}/{len(scores)}={task_complete/len(scores)}"
)
# print(f"All the cost is {sum(all_costs)}")
print(f"The average time consuming is {sum(all_times)/len(all_times)}")
print(
    f"The performance is {sum(scores)/len(scores)} with {len(scores)} out of 74 tasks scored."
)
