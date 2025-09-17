from pydantic import BaseModel, Field
from typing import Optional, Literal
from uuid import UUID


class CreateUserParam(BaseModel):
    username: str = Field(description="Define user name or the task type.")


class CreateSessionConfig(BaseModel):
    reset_llm_config: Optional[dict] = None
    reset_code_executor: Optional[dict] = None
    tool_mode: Literal["default", "dsbench", "datamodeling"] = "default"


class UserSessionConfig(BaseModel):
    user_id: UUID = Field(..., description="The ID of the user")
    session_id: UUID = Field(..., description="The ID of the session")


class ChatParam(BaseModel):
    # user_session_config: UserSessionConfig, query: str = Body(...)
    # TODO
    user_id: UUID = Field(..., description="The ID of the user")

    session_id: UUID = Field(..., description="The ID of the session")
    query: str = Field(..., description="The user query to process")

    # 可选参数
    agent_config: Optional[dict] = Field(
        None, description="Configuration for the agent"
    )
    planning_max_number: Optional[int] = Field(
        None, description="Maximum number of planning iterations"
    )
    execution_max_number: Optional[int] = Field(
        None, description="Maximum number of execution iterations"
    )
    work_mode: Literal["jupyter", "jupyter+script"] = Field(
        default="jupyter", description="working mode of datawise agent."
    )


class PlanConfig(BaseModel):
    planning: bool = Field(True, description="Enable or disable planning")
    planning_max_number: Optional[int] = Field(
        None, description="Maximum number of planning iterations"
    )


class ExecutionConfig(BaseModel):
    # append_code: bool = Field(True, description="Enable or disable code appending")  # 如果需要，可以取消注释
    execution_max_number: Optional[int] = Field(
        None, description="Maximum number of execution iterations"
    )


class DebugConfig(BaseModel):
    self_debug: bool = Field(True, description="Enable or disable self debugging")
    debug_max_number: Optional[int] = Field(
        None, description="Maximum number of debugging iterations"
    )


class DatawiseAgentConfig(BaseModel):
    plan: PlanConfig = Field(
        default_factory=PlanConfig, description="Configuration for planning"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Configuration for execution"
    )
    debug: DebugConfig = Field(
        default_factory=DebugConfig, description="Configuration for debugging"
    )
    evaluation: bool = Field(True, description="Enable or disable evaluation")

    max_step_number: Optional[int] = Field(
        default=None, description="max step number of Planning and Execution."
    )
    max_debug_by_step: Optional[int] = Field(
        default=None, description="max debug failure number in one step."
    )


class CodeExecutorConfig(BaseModel):
    use_docker: bool = Field(default=True)
    use_proxy: bool = Field(default=True)
    use_gpu: bool = Field(default=False)
    image_name: str = Field(default="my-jupyter-image")


class SessionInfo(BaseModel):
    session_id: UUID
    session_name: Optional[str] = None


class UserInfo(BaseModel):
    user_id: UUID
    user_name: Optional[str] = None
