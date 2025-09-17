import uuid
import httpx
import websockets
import asyncio
from pathlib import Path
from typing import Optional, Literal

# 配置服务器地址
BASE_URL = "http://0.0.0.0:8000"
BASE_WS_URL = "ws://localhost:8000/register_websocket"


async def register_websocket(user_id: str, session_id: str):
    ws_url = f"{BASE_WS_URL}/{user_id}/{session_id}"
    async with websockets.connect(ws_url) as websocket:
        print(f"Registered WebSocket for {user_id}/{session_id}")
        while True:
            message = await websocket.recv()
            print(f"Received message from {session_id}: {message}")


# 创建一个用户
def create_user(username: str):
    response = httpx.post(
        f"{BASE_URL}/create_user/", json={"username": username}, timeout=None
    )
    response_data = response.json()
    print("Create User Response:", response_data)
    return response_data["user_id"]


# 创建一个会话
def create_session(
    user_id: str,
    session_name: str,
    tool_mode: Literal["default", "dsbench", "datamodeling"] = "default",
):
    response = httpx.post(
        f"{BASE_URL}/users/{user_id}/sessions/{session_name}",
        json={
            "tool_mode": tool_mode,
        },
        timeout=None,
    )
    response_data = response.json()
    print("Create Session Response:", response_data)
    return response_data["session_id"]


# 聊天
def chat(
    user_id: str,
    session_id: str,
    query: str,
    work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
):
    response = httpx.post(
        f"{BASE_URL}/chat/",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "work_mode": work_mode,
            # 可选参数可以省略
            # "agent_config": {...},
        },
        timeout=3600,
    )
    response_data = response.json()
    print("Chat Response:", response_data)
    return response_data


# 上传文件
def upload_file(
    user_id: str,
    session_id: str,
    file_path: str,
    filename_to_save: Optional[str] = None,
):
    with open(file_path, "rb") as f:
        # filename = Path(file_path).name
        if filename_to_save == None:
            filename = Path(file_path).name
        else:
            filename = filename_to_save

        files = {"files": (filename, f)}

        response = httpx.post(
            f"{BASE_URL}/upload/",
            data={"user_id": user_id, "session_id": session_id},
            files=files,
            timeout=None,
        )
    response_data = response.json()
    print("Upload File Response:", response_data)


async def main():
    # asyncio.run(main())
    user_id = "test_user_id"  # 替换为真实的用户 ID
    session_ids = ["session1", "session2", "session3"]  # 替换为实际会话

    tasks = [register_websocket(user_id, session_id) for session_id in session_ids]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # 创建用户和会话
    user_id = create_user("test_user")

    import pdb

    pdb.set_trace()

    # 发送一个查询

    query = """You are solving this machine learning tasks of time series classification: 
The dataset presented here (the Ethanol Concentration dataset) comprises real-world time series data. We have splitted the dataset into three parts of train, valid and test. The input is a sequence of observed features (INPUT_SEQ_LEN=1751, INPUT_DIM=3). Your task is to predict the labels for the given sequence, where the label is in range of {0, 1, 2, 3}. The evaluation metric is accuracy.
We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a time series classification model to get a good performance on the given fixed sequences.
In this task, you should work in `env/` folder, which means the working memory when executing any code should be in `env/`. So change the working memeory to `env` FIRST before you start your work. You should try your best to complete the task and optimize the performance.

"""
    # session_id = create_session(user_id, "test_session")
    # chat(user_id, session_id, query)

    query = "Download the file from the following link: https://github.com/Luffyzm3D2Y/CHEF_assignment_for_NLP/raw/refs/heads/main/test.csv. This is a phenotype file related to patients. Please first review the contents of the file, then investigate whether there are significant differences in survival rates and infection rates among different age groups."
    session_id = create_session(user_id, "test_session")
    chat(user_id, session_id, query)

    query = "download data from https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv and plot a visualization that tells us about the relationship between weight and horsepower. Save the plot to a file. Print the fields in a dataset before visualizing it."
    session_id = create_session(user_id, "test_session")
    chat(user_id, session_id, query)

    query = """从以下链接下载文件：`https://github.com/Luffyzm3D2Y/CHEF_assignment_for_NLP/raw/refs/heads/main/test.csv`。 这是一个和病人相关的表型文件。请你先了解下文件内容，然后探究一下不同年龄段的人群在生存率和感染率上是否存在显著差异。你需要使用中文回答我的问题。"""
    chat(user_id, session_id, query)

    # 文件上传测试
    # upload_file(user_id, session_id, "test.csv")

    # 交互式输入查询
    while True:
        user_query = input("Input your query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        chat(user_id, session_id, user_query)
