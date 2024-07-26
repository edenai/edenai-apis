from typing import Dict, List, Union, Generator, Optional
import json
import base64
import requests
import httpx
from edenai_apis.features.multimodal.chat import (
    ChatDataClass,
    StreamChat,
    ChatMessageDataClass,
    ChatStreamResponse,
)
from edenai_apis.features.multimodal.multimodal_interface import MultimodalInterface
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException
from edenai_apis.apis.google.google_helpers import calculate_usage_tokens


class GoogleMultimodalApi(MultimodalInterface):

    @staticmethod
    def __format_google_messages(
        messages: List[ChatMessageDataClass],
    ) -> List[Dict[str, str]]:
        """
        Format messages into a format accepted by Google Gemini vision.

        Args:
            messages (List[ChatMessageDataClass]): List of messages to be formatted.

        Returns:
            List[Dict[str, str]]: Transformed messages in Google Gemini accepted format.

        >>> Accepted format:
        [
            {
            "role": string,
            "parts": [
                {
                // Union field data can be only one of the following:
                "text": {"text" : string},
                "inlineData": {
                    "mimeType": string,
                    "data": string
                }
                }
            ]
            }
        ]

        """
        transformed_messages = []
        for message in messages:
            role = message["role"]
            if role == "user":
                parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append({"text": content["content"]["text"]})
                    elif content["type"] == "media_url":
                        media_url = content["content"]["media_url"]
                        media_type = content["content"]["media_type"]
                        response = httpx.get(media_url)
                        data = base64.b64encode(response.content).decode("utf-8")
                        parts.append(
                            {"inline_data": {"data": data, "mime_type": media_type}}
                        )
                    elif content["type"] == "media_base64":
                        media_base64 = content["content"]["media_base64"]
                        parts.append(
                            {
                                "inline_data": {
                                    "data": media_base64,
                                    "mime_type": content["content"]["media_type"],
                                }
                            }
                        )
                transformed_messages.append({"role": role, "parts": parts})
            elif role == "assistant":
                transformed_messages.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": message.get("content")[0]
                                .get("content")
                                .get("text"),
                            }
                        ],
                    }
                )
        return transformed_messages

    @staticmethod
    def __chat_stream_generator(response: requests.Response) -> Generator:
        for resp in response.iter_lines():
            raw_data = resp.decode()
            if "data: " in raw_data:
                _, content = raw_data.split("data :")
                try:
                    content_json = json.loads(content)
                    yield ChatStreamResponse(
                        text=content_json["candidates"][0]["content"]["parts"][0][
                            "text"
                        ],
                        blocked=False,
                        provider="google",
                    )
                except Exception as exc:
                    return

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
        response_format = None,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        api_key = self.api_settings.get("genai_api_key")
        base_url = "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        url = base_url.format(model=model, api_key=api_key)
        formatted_messages = self.__format_google_messages(messages=messages)
        payload = {
            "contents": formatted_messages,
            "generationConfig": {
                "candidateCount": 1,
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
                "topK": top_k,
                "stopSequences": stop_sequences,
            },
        }
        if chatbot_global_action:
            payload["system_instruction"] = (
                {"parts": {"text": chatbot_global_action}},
            )

        if stream is False:
            response = requests.post(url, json=payload)
            try:
                original_response = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException(
                    "An error occurred while parsing the response."
                ) from exc

            if response.status_code != 200:
                raise ProviderException(
                    message=original_response["error"]["message"],
                    code=response.status_code,
                )
            generated_text = original_response["candidates"][0]["content"]["parts"][0][
                "text"
            ]
            calculate_usage_tokens(original_response=original_response)
            standardized_response = ChatDataClass.generate_standardized_response(
                generated_text=generated_text,
                messages=messages,
            )

            return ResponseType[ChatDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )
        else:
            url.replace("generateContent", "streamGenerateContent?alt=sse")
            response = requests.post(url, json=payload, stream=True)
            try:
                original_response = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException(
                    "An error occurred while parsing the response."
                ) from exc

            if response.status_code != 200:
                raise ProviderException(
                    message=response.text,
                    code=response.status_code,
                )
            stream_response = self.__chat_stream_generator(response)
            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=stream_response),
            )
