"""
main.py
=======

Backend server for DatawiseAgent built with FastAPI.

This server provides RESTful APIs and WebSocket endpoints for:
- User creation and session management
- Chatting with the agent
- Uploading and processing files
- Querying session data and evaluation logs

⚠️ Note:
This repository only open-sources the backend server.
The WebSocket endpoint at `/register_websocket/{user_id}/{session_id}`
is reserved for frontend integration (not provided here).
"""

import dotenv

dotenv.load_dotenv()
import uuid
from uuid import UUID
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Body,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List
from pathlib import Path
import json
import os
import asyncio
from starlette.websockets import WebSocketState

from datawiseagent.memory.session import SessionContent, CellHistoryMemory
from datawiseagent.agents.datawise_agent import (
    DatawiseAgent,
    global_config,
    llm_config,
    agent_config,
    code_executor_config,
    SESSIONS_LOG_PATH,
)
from datawiseagent.management.manager import DatawiseAgentManager
from datawiseagent.common.types import (
    CreateSessionConfig,
    CreateUserParam,
    UserSessionConfig,
    ChatParam,
    SessionInfo,
)


app = FastAPI()
agent_manager = DatawiseAgentManager()


@app.post("/create_user/")
async def create_user(param: CreateUserParam):
    username = param.username
    user_id = agent_manager.create_user(username)

    return {"user_id": str(user_id), "username": username}


@app.post("/users/{user_id}/sessions/{session_name}")
async def create_session(user_id: UUID, session_name: str, config: CreateSessionConfig):

    agent = agent_manager.get_agent(user_id)

    if config.reset_llm_config == None:
        config.reset_llm_config = llm_config

    if config.reset_code_executor == None:
        config.reset_code_executor = code_executor_config

    session_id = await agent.create_new_session(
        config.reset_llm_config,
        config.reset_code_executor,
        session_name,
        config.tool_mode,
    )
    return {"session_id": str(session_id), "session_name": session_name}


@app.post("/chat/")
async def chat(param: ChatParam):
    param_dict = param.model_dump(exclude_unset=True)
    user_id = param.user_id
    session_id = param.session_id

    agent = agent_manager.get_agent(user_id)
    if session_id not in agent.sessions:
        return {"error": "Invalid session ID"}
    if "query" in param_dict:
        param_dict.pop("user_id", None)

    last_user_content, last_response_content = await agent.chat(**param_dict)
    return {
        "user_content": last_user_content,
        "response_content": last_response_content,
    }


@app.post("/upload/")
async def upload_file(
    user_id: UUID = Form(...),
    session_id: UUID = Form(...),
    files: List[UploadFile] = File(...),
):
    agent = agent_manager.get_agent(user_id)
    if session_id not in agent.sessions:
        return {"error": "Invalid session ID"}

    paths = await agent.process_uploaded_files(session_id, files)

    return {"message": "Files processed successfully", "results": paths}


@app.get("/users", response_model=List[str])
async def list_users():
    user_dir = Path(SESSIONS_LOG_PATH) / "users"
    if not user_dir.exists():
        return []
    user_ids = [d.name for d in user_dir.iterdir() if d.is_dir()]
    return user_ids


@app.get("/users/{user_id}/sessions", response_model=List[SessionInfo])
async def list_sessions(user_id: UUID):
    """
    List all sessions for a given user.

    Args:
        user_id (UUID): The unique identifier of the user.

    Returns:
        List[dict]: A list of dictionaries containing session_id and session_name.
    """
    session_dir = Path(SESSIONS_LOG_PATH) / "users" / str(user_id) / "sessions"

    if not session_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"User {user_id}'s sessions not found"
        )
    sessions = []
    for session_id_dir in session_dir.iterdir():
        if session_id_dir.is_dir():
            session_id = session_id_dir.name
            # Find all JSON files in the session_id directory
            json_files = list(session_id_dir.glob("*.json"))
            if not json_files:
                continue  # Skip if no JSON files are found

            # Assuming there's only one session_name.json per session_id directory
            session_file = json_files[0]
            session_name = session_file.stem  # Extract filename without extension

            sessions.append(
                SessionInfo(session_id=session_id, session_name=session_name)
            )

    return sessions


@app.get("/users/{user_id}/sessions/{session_id}/data", response_model=SessionContent)
async def get_session_data(user_id: UUID, session_id: UUID):
    session_id_dir: Path = (
        Path(SESSIONS_LOG_PATH) / "users" / str(user_id) / "sessions" / str(session_id)
    )
    if not session_id_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id}'s session {session_id} directory not found!",
        )
    json_files = list(session_id_dir.glob("*.json"))
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id}'s session {session_id} JSON file not found!",
        )
    # Assuming there's only one session_name.json per session_id directory
    session_file = json_files[0]

    with open(session_file, "r") as f:
        session_data = json.load(f)

    try:
        validated_session = SessionContent.from_json(session_data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    return validated_session


@app.websocket("/register_websocket/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: UUID, session_id: UUID):
    """
    WebSocket endpoint for real-time communication between frontend and backend.

    Note:
        - This is a reserved interface intended for a frontend client (not open-sourced here).
        - Used to stream messages and keep sessions alive with ping/pong.
        - Backend users typically do not need to call this directly.
    """
    await websocket.accept()
    agent_manager.add_websocket(user_id, session_id, websocket)

    async def send_ping():
        while True:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ping"})
            await asyncio.sleep(10)

    ping_task = asyncio.create_task(send_ping())

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {user_id}/{session_id}: {data}")
    except WebSocketDisconnect:
        agent_manager.remove_websocket(user_id, session_id, websocket)
    finally:
        ping_task.cancel()


# CORS configuration (only needed if you want to serve a web frontend)
# origins = ["http://localhost:3000"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
