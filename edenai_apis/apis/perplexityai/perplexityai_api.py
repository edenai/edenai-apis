from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatStreamResponse
from typing import Dict, List, Literal, Optional, Union, Generator
import requests
import json


class PerplexityApi(ProviderInterface, TextInterface):
    provider_name = "perplexityai"

    def __init__(self, api_keys: Dict = {}):
        self.url = "https://api.perplexity.ai"

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_settings['api_key']}",
        }

    @staticmethod
    def __text_to_json(
        lst_data: List[str],
    ) -> Generator[ChatStreamResponse, None, None]:
        length = len(lst_data)
        i = 0
        while i < length:
            if lst_data[i].startswith("data:"):
                lst_data[i] = lst_data[i].replace("data: ", "")
                lst_data[i] = lst_data[i].replace("\r", "")
                i += 1
            else:
                lst_data.pop(i)
            length = len(lst_data)
        for token in lst_data:
            jsonres = json.loads(token)
            if error := jsonres.get("error"):
                raise ProviderException(error.get("message"), error.get("code") or 400)
            yield ChatStreamResponse(
                text=jsonres["choices"][0]["delta"]["content"],
                blocked=False,
                provider="perplexityai",
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
        messages = []

        if any([available_tools, tool_results]):
            raise ProviderException("This provider does not support the use of tools")

        if chatbot_global_action:
            messages.append({"role": "system", "content": chatbot_global_action})

        if previous_history:
            for message in previous_history:
                messages.append(
                    {"role": message.get("role"), "content": message.get("message")},
                )

        messages.append({"role": "user", "content": text})
        url = f"{self.url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)
        else:
            if not stream:
                try:
                    original_response = response.json()
                except requests.JSONDecodeError as exp:
                    raise ProviderException(
                        response.text, code=response.status_code
                    ) from exp

                generated_text = original_response["choices"][0]["message"]["content"]
                message = [
                    ChatMessageDataClass(role="user", message=text),
                    ChatMessageDataClass(role="system", message=generated_text),
                ]
                standardized_response = ChatDataClass(
                    generated_text=generated_text, message=message
                )

                return ResponseType[ChatDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                )
            else:
                data = response.text
                lst_data = data.split("\n")
                return ResponseType[StreamChat](
                    original_response=None,
                    standardized_response=StreamChat(
                        stream=self.__text_to_json(lst_data)
                    ),
                )
