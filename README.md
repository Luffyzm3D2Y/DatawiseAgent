
# Initialize Environment
## Conda
conda create -n datawise python=3.10 -y
conda activate datawise
pip install -r requirements.txt

## Docker
As we recommend using Docker sandbox in DatawiseAgent, we need to preinstall Docker, and then build code executor image as needed.
In specify,
if there's no need for GPU computing, build `my-jupyter-image` and specify `image_name: my-jupyter-image` in config yaml file (as see [Configuration](## Configuration)):

```
docker build -t my-jupyter-image -f datawiseagent/coding/jupyter/default_jupyter_server.dockerfile . --progress=plain
```

if there's need for GPU computing (such as for DataModeling tasks), build `my-jupyter-image-gpus` and specify `image_name: my-jupyter-image-gpus` in config yaml file (as see [Configuration](## Configuration)):

```
docker build -t my-jupyter-image-gpus -f datawiseagent/coding/jupyter/cuda_jupyter_server.dockerfile . --progress=plain
```

## Vllm
Our agent framework demonstrates strong effectiveness, robustness and adaptiveness across various LLMs, including GPT-4o, GPT-4o mini, and Qwen2.5 7B, 14B, 32B, 72B. To support deployment for open-source LLMs, we recommend use [vllm](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) to deploy open-sourced LLMs as an OpenAI-Compatible Server. After that, it works well by justing setting `OPENAI_API_KEY` and `OPENAI_BASE_URL`.



# Start DatawiseAgent as a Backend Server

## Configuration
1. configure `.env`
copy `.env.example` to `.env`, and fill all the neccessary env variables, including `OPENAI_API_KEY` and `OPENAI_BASE_URL`. In this repo, we recommend starting DatawiseAgent as a backend server, and the default configuaration is configured by the file `configs/default_config.yaml`. If you want to customize a different config file, create a new yaml file and specify it in the `.env` by `export CUSTOM_CONFIG=`.

2. configure `configs/*.yaml`
The yaml files in `configs/` specify all the hyperparameters of DatawiseAgent, with each field explained in annotations.

To facilitate reproducing our experimental results in our paper, we save 2 config files in `configs/`:
- `default_config.yaml`: InfiAgentBench, MatplotBench
- `datamodeling.yaml`: Datamodeling tasks from DSBench

The main difference between them is the docker image used and some minor modification in limitation to the finite-state transducer of DatawiseAgent.

## Start a Server

start DatawiseAgent as a backend server which supports multi-user and multi-session scenarios.
python main.py

http://localhost:8000


# Evaluation

## Experiment results
We evaluate DatawiseAgent on three representa111 tive data science scenarios, namely data analysis, scientific visualization, and predictive modeling,
across both proprietary (GPT-4o, GPT-4o mini) and open-source (Qwen2.5 at multiple scales) LLMs. The experiment results are as follows:
[插入两张图片，竖着放，分别是`evaluation/experimental_results/effective_result.jpg`, `evaluation/experimental_results/robust_result.jpg`]


To facilitate further research, we open-source the primary experiement results and agent trajectories in `evaluation/experimental_results`.
Also we can reproduce our results following the following steps.

All the evaluations should at the root of `evaluation/`.
```
cd evaluation/
``

## Data Analysis [InfiAgentBench]


## Scientific Visualizetion [MatplotBench]


## DataModeling [DSBench]

