from typing import Dict, List, Literal, Optional, Union
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.llmengine import LLMEngine


class DeepseekApi(ProviderInterface, TextInterface):
    provider_name = "deepseek"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.llm_client = LLMEngine(
            provider_name="together_ai", provider_config={"api_key": self.api_key}
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str] = None,
        previous_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stream: bool = False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
            text=text,
            chatbot_global_action=chatbot_global_action,
            previous_history=previous_history,
            temperature=temperature,
            max_tokens=max_tokens,
            model=f"deepseek-ai/{model}",
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
            **kwargs,
        )
        return response
