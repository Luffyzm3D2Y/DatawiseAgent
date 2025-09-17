from datawiseagent.agents.datawise_agent import DatawiseAgent
from uuid import UUID, uuid4
from typing import Optional, List, Tuple
from fastapi import WebSocket
from datawiseagent.memory.session import SessionContent
from datawiseagent.common.types import SessionInfo

class DatawiseAgentManager:
    def __init__(self):
        self.agents: dict[UUID, DatawiseAgent] = {}  # 存储每个用户的DatawiseAgent实例

        # store active sessions' websockets
        # multiple clients 2 one server
        self.active_websockets: dict[Tuple[UUID, UUID], List[WebSocket]] = {}

    def get_agent(self, user_id: UUID) -> DatawiseAgent:
        """
        Fetch user's agent instance. If there's no agent instance, create one.
        """

        if user_id not in self.agents:
            self.agents[user_id] = DatawiseAgent(user_id=user_id, agent_manager=self)

        return self.agents[user_id]

    def create_user(self, user_name: Optional[str] = None) -> UUID:
        user_id = uuid4()
        self.agents[user_id] = DatawiseAgent(user_id, user_name, agent_manager=self)
        return user_id

    def add_websocket(self, user_id: UUID, session_id: UUID, websocket: WebSocket):
        """
        将新的 WebSocket 连接添加到 active_websockets 中。
        """
        key = (user_id, session_id)
        if key not in self.active_websockets:
            self.active_websockets[key] = []

        self.active_websockets[key].append(websocket)

    def remove_websocket(self, user_id: UUID, session_id: UUID, websocket: WebSocket):
        """
        从 active_websockets 中移除指定的 WebSocket 连接。
        """
        key = (user_id, session_id)
        if key in self.active_websockets:
            if websocket in self.active_websockets[key]:
                self.active_websockets[key].remove(websocket)
                # 如果该会话已没有活跃的 WebSocket，删除该键
                if not self.active_websockets[key]:
                    del self.active_websockets[key]

    def get_websockets(self, user_id: UUID, session_id: UUID) -> List[WebSocket]:
        """
        获取指定用户和会话的所有活跃 WebSocket 连接。
        """
        key = (user_id, session_id)
        return self.active_websockets.get(key, [])

    def load_all_sessions(self):
        pass

    async def broadcast_session_update(
        self, user_id: UUID, session_id: UUID, session_content: SessionContent
    ):
        websockets = self.get_websockets(user_id, session_id)

        data = session_content.model_dump_json()
        for websocket in websockets:
            try:
                await websocket.send_text(data)
            except Exception as e:
                # 如果发送失败，移除该 WebSocket 连接
                self.remove_websocket(user_id, session_id, websocket)
