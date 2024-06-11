from typing import Dict, List, Union, Generator, Optional
import json
import requests
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import base64
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
                "role": "user",
                "parts": [
                    "can you describe this image ? ",
                    {"mime_type": "image/jpeg", "data": base64_image},
                ],
            }
        ]
        """
        formatted_messages = []

        for message in messages:
            formatted_message = {"role": message.get("role"), "parts": []}
            for content_item in message.get("content"):
                if content_item["type"] == "text":
                    formatted_message["parts"].append(content_item["content"]["text"])
                elif content_item["type"] == "media_base64":
                    formatted_message["parts"].append(
                        {
                            "mime_type": content_item["content"]["media_type"],
                            "data": content_item["content"]["media_base64"],
                        }
                    )
                elif content_item["type"] == "media_url":
                    media_url = content_item["content"]["media_url"]
                    media_type = content_item["content"]["media_type"]
                    response = requests.get(media_url)
                    data = base64.b64encode(response.content).decode("utf-8")
                    formatted_message["parts"].append(
                        {"mime_type": media_type, "data": data}
                    )

            formatted_messages.append(formatted_message)

        return formatted_messages

    @staticmethod
    def __chat_stream_generator(response: requests.Response) -> Generator:
        for chunk in response:
            yield ChatStreamResponse(text=chunk.text, blocked=False, provider="google")

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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        formatted_messages = self.__format_google_messages(messages=messages)
        generation_config = {
            "temperature": temperature,
            "top_p": top_k,
            "top_k": top_p,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop_sequences,
        }
        model = genai.GenerativeModel(
            model,
            generation_config=generation_config,
            system_instruction=chatbot_global_action,
        )

        if stream is False:
            try:
                response = model.generate_content(formatted_messages)
            except GoogleAPIError as exc:
                raise ProviderException(exc.message) from exc
            try:
                original_response = response.to_dict()
            except json.JSONDecodeError as exc:
                raise ProviderException(
                    "An error occurred while parsing the response."
                ) from exc

            generated_text = response.text
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
            try:
                response = model.generate_content(formatted_messages, stream=True)
            except GoogleAPIError as exc:
                raise ProviderException(exc.message) from exc

            stream_response = self.__chat_stream_generator(response)
            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=stream_response),
            )
