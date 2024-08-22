import itertools
from typing import Dict, List, Literal, Optional, Union, Generator
import requests
import json
from edenai_apis.apis.openai.helpers import convert_tools_to_openai

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ToolCall,
)
from edenai_apis.features.text.chat.helpers import get_tool_call_from_history_by_id
from edenai_apis.features.text.embeddings import EmbeddingDataClass, EmbeddingsDataClass
from edenai_apis.features.text.generation.generation_dataclass import (
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class MistralApi(ProviderInterface, TextInterface):
    provider_name = "mistral"

    def __init__(self, api_keys: Dict = {}, **kwargs) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.mistral.ai/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def __get_stream_response(self, response: requests.Response) -> Generator:
        """returns a generator of chat messages

        Args:
            response (requests.Response): The post request

        Yields:
            Generator: generator of messages
        """
        for res in response.iter_lines():
            chunk = res.decode().split("data: ")
            if len(chunk) > 1:
                if chunk[1] != "[DONE]":
                    data = json.loads(chunk[1])
                    yield ChatStreamResponse(
                        text=data["choices"][0]["delta"]["content"],
                        blocked=not data["choices"][0].get("finish_reason")
                        in (None, "stop"),
                        provider=self.provider_name,
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
        previous_history = previous_history or []
        messages = []
        for msg in previous_history:
            message = {
                "role": msg.get("role"),
                "content": msg.get("message"),
            }
            if msg.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": t.get("id"),
                        "function": {
                            "name": t.get("name"),
                            "arguments": t.get("arguments"),
                        },
                    }
                    for t in msg["tool_calls"]
                ]
            messages.append(message)

        if text and not tool_results:
            messages.append({"role": "user", "content": text})

        if tool_results:
            for tool in tool_results or []:
                tool_call = get_tool_call_from_history_by_id(
                    tool["id"], previous_history
                )
                try:
                    result = json.dumps(tool["result"])
                except json.JSONDecodeError:
                    result = str(result)
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "name": tool_call["name"],
                    }
                )

        if chatbot_global_action:
            messages.insert(0, {"role": "system", "content": chatbot_global_action})

        payload = {
            "model": f"{self.provider_name}-{model}",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if available_tools:
            payload["tools"] = convert_tools_to_openai(available_tools)
            payload["tool_choice"] = "any" if tool_choice == "required" else tool_choice

        if not stream:
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

            # Build a list of ChatMessageDataClass objects for the conversation history
            message = original_response["choices"][0]["message"]
            generated_text = message["content"]
            original_tools_calls = message.get("tool_calls") or []
            tool_calls = []
            for tool_call in original_tools_calls:
                tool_calls.append(
                    ToolCall(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    )
                )

            message = [
                ChatMessageDataClass(role="user", message=text, tools=available_tools),
                ChatMessageDataClass(
                    role="assistant", message=generated_text, tool_calls=tool_calls
                ),
            ]

            # Build the standardized response
            standardized_response = ChatDataClass(
                generated_text=generated_text, message=message
            )

            # Calculate number of tokens :
            original_response["usage"]["total_tokens"] = (
                original_response["usage"]["completion_tokens"]
                + original_response["usage"]["prompt_tokens"]
            )

            return ResponseType[ChatDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )
        else:
            payload["stream"] = True
            response = requests.post(
                self.url + "v1/chat/completions",
                json=payload,
                headers=self.headers,
                stream=True,
            )
            response = self.__get_stream_response(response)
            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=response),
            )

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
