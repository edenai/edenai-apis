from typing import Dict, List, Literal, Union, Optional
from edenai_apis.features import ProviderInterface, TextInterface, ImageInterface
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoDetectionDataClass,
)
from edenai_apis.features.text import SummarizeDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatDataClass,
)
from edenai_apis.features.multimodal.chat.chat_dataclass import (
    ChatDataClass as ChatMultimodalDataClass,
    StreamChat as StreamChatMultimodal,
    ChatMessageDataClass as ChatMultimodalMessageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.llmengine.llm_engine import LLMEngine


class AnthropicApi(ProviderInterface, TextInterface, ImageInterface):
    provider_name = "anthropic"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.llm_client = LLMEngine(
            provider_name=self.provider_name,
            provider_config={"api_key": self.api_settings.get("api_key")},
        )

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[SummarizeDataClass]:
        response = self.llm_client.summarize(
            text=text,
            model=model,
            **kwargs,
        )
        return response

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
            previous_history=previous_history,
            chatbot_global_action=chatbot_global_action,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
            **kwargs,
        )
        return response

    def multimodal__chat(
        self,
        messages: List[ChatMultimodalMessageDataClass],
        chatbot_global_action: Optional[str],
        temperature: float = 0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        stream: bool = False,
        provider_params: Optional[dict] = None,
        response_format=None,
    ) -> ResponseType[Union[ChatMultimodalDataClass, StreamChatMultimodal]]:
        response = self.llm_client.multimodal_chat(
            messages=messages,
            chatbot_global_action=chatbot_global_action,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stop_sequences=stop_sequences,
            top_k=top_k,
            top_p=top_p,
            stream=stream,
            response_format=response_format,
        )
        return response

    def image__logo_detection(
        self,
        file: str,
        file_url: str = "",
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[LogoDetectionDataClass]:
        response = self.llm_client.logo_detection(
            file=file,
            file_url=file_url,
            model=model,
            **kwargs,
        )
        return response
