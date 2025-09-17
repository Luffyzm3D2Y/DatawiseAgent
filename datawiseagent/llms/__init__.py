from datawiseagent.common.registry import Registry

llm_registry = Registry(name="LLMRegistry")
LOCAL_LLMS = {
    # <model name> : <openai_compatible>
    "Llama-3-8B-8192-3-epoch": {
        "openai_compatible": False,
        "response_format": True,
    },
    "/zyxx0629001/checkpoints/functionary-medium-v3.0": {
        "openai_compatible": True,
        "response_format": False,
    },
}
LOCAL_LLMS_MAPPING = {
    # "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    # "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    # "llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",
    # "vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
    # "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
    # "Llama-3-8B-8192-1-epoch": "meta-llama/Meta-Llama-3-8B",
    # "Llama-3-8B-8192-2-epoch": "meta-llama/Meta-Llama-3-8B",
    "Llama-3-8B-8192-3-epoch": "meta-llama/Meta-Llama-3-8B",
    "/zyxx0629001/checkpoints/functionary-medium-v3.0": "functionary/medium",
}

from .base import BaseLLM, BaseChatModel, BaseCompletionModel
from .openai import OpenAIChat


def load_llm(llm_config: dict):
    """
    llm_config example:

        llm_type: openai-chat
        # model: gpt-4o
        # model: gpt-4o-mini
        # model: gpt-4-1106-preview

        # openai_compatible: false
        temperature: 0.1

    """
    llm_type = llm_config.pop("llm_type", "text-davinci-003")
    return llm_registry.build(llm_type, **llm_config)
