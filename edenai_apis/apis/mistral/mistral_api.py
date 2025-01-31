import itertools
from typing import Dict, List, Literal, Optional, Union, Generator
import requests
import json

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ToolCall,
)
from edenai_apis.features.text.embeddings import EmbeddingDataClass, EmbeddingsDataClass
from edenai_apis.features.text.generation.generation_dataclass import (
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.llm_engine.llm_engine import LLMEngine


class MistralApi(ProviderInterface, TextInterface):
    provider_name = "mistral"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.mistral.ai/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str
    ) -> ResponseType[GenerationDataClass]:
        messages = [
            {
                "role": "system",
                "content": "Act as Text Generator, complete the given texts.",
            },
            {"role": "user", "content": text},
        ]

        payload = {
            "model": f"{self.provider_name}-{model}",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            self.url + "v1/chat/completions", json=payload, headers=self.headers
        )
        try:
            original_response = response.json()
            if "message" in original_response or response.status_code >= 400:
                message_error = original_response["message"]
                raise ProviderException(message_error, code=response.status_code)
        except Exception:
            raise ProviderException(response.text, code=response.status_code)

        generated_text = original_response["choices"][0]["message"]["content"]

        # Calculate number of tokens :
        original_response["usage"]["total_tokens"] = (
            original_response["usage"]["completion_tokens"]
            + original_response["usage"]["prompt_tokens"]
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(generated_text=generated_text),
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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
            text=text,
            chatbot_global_action=chatbot_global_action,
            previous_history=previous_history,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
        )
        return response

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model.split("__")[1]
        payload = {"model": model, "input": texts}
        response = requests.post(
            url=self.url + "v1/embeddings", json=payload, headers=self.headers
        )
        try:
            original_response = response.json()
            if "message" in original_response or response.status_code >= 400:
                message_error = original_response["message"]
                raise ProviderException(message_error, code=response.status_code)
        except Exception:
            raise ProviderException(response.text, code=response.status_code)

        items = []
        embeddings = original_response["data"]

        for embedding in embeddings:
            items.append(EmbeddingDataClass(embedding=embedding["embedding"]))

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=EmbeddingsDataClass(items=items),
        )
