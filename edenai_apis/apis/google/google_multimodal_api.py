from typing import List, Union, Optional
from edenai_apis.features.multimodal.chat import (
    ChatDataClass,
    StreamChat,
    ChatMessageDataClass,
)
from edenai_apis.features.multimodal.multimodal_interface import MultimodalInterface
from edenai_apis.utils.types import ResponseType


class GoogleMultimodalApi(MultimodalInterface):

    def multimodal__chat(
        self,
        messages: List[ChatMessageDataClass],
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
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.clients["llm_client"].multimodal_chat(
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
            **kwargs,
        )
        return response
