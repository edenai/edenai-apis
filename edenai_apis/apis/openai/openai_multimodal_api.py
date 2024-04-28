from typing import Dict, List, Union, Optional
import openai
from edenai_apis.features import MultimodalInterface
from edenai_apis.features.multimodal.chat import (
    ChatDataClass,
    StreamChat,
    ChatMessageDataClass,
    ChatStreamResponse,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException


class OpenaiMultimodalApi(MultimodalInterface):

    @staticmethod
    def __format_openai_messages(
        messages: List[ChatMessageDataClass],
    ) -> List[Dict[str, str]]:
        """
        Format messages into a format accepted by OpenAI.

        Args:
            messages (List[ChatMessageDataClass]): List of messages to be formatted.

        Returns:
            List[Dict[str, str]]: Transformed messages in OpenAI accepted format.

        >>> Accepted format:
            [
                {
                    "role": <role>,
                    "content": [
                        {
                            "type": "text",
                            "text": <text_content>
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": <image_url>}
                        }
                    ]
                }
            ]

        """
        transformed_messages = []
        for message in messages:
            if message["role"] == "user":
                transformed_message = {"role": message["role"], "content": []}
                for item in message["content"]:
                    if item["type"] == "text":
                        transformed_message["content"].append(
                            {"type": "text", "text": item.get("content").get("text")}
                        )
                    elif item["type"] == "media_url":
                        transformed_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": item["content"]["media_url"]},
                            }
                        )
                    else:
                        b64_data = item.get("content").get("media_base64")
                        media_data_url = f"data:image/jpeg;base64,{b64_data}"
                        transformed_message["content"].append(
                            {"type": "image_url", "image_url": {"url": media_data_url}}
                        )
            else:
                transformed_message = {
                    "role": message["role"],
                    "content": message.get("content")[0].get("content").get("text"),
                }

            transformed_messages.append(transformed_message)

        return transformed_messages

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
        formatted_messages = self.__format_openai_messages(messages)

        if chatbot_global_action:
            formatted_messages.insert(
                0, {"role": "system", "content": chatbot_global_action}
            )

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        try:
            response = openai.ChatCompletion.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        if stream is False:
            generated_text = response["choices"][0]["message"]["content"]

            standardized_response = ChatDataClass.generate_standardized_response(
                generated_text=generated_text, messages=messages
            )

            return ResponseType[ChatDataClass](
                original_response=response,
                standardized_response=standardized_response,
            )

        else:
            stream = (
                ChatStreamResponse(
                    text=chunk["choices"][0]["delta"].get("content", ""),
                    blocked=not chunk["choices"][0].get("finish_reason")
                    in (None, "stop"),
                    provider="openai",
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response=None, standardized_response=StreamChat(stream=stream)
            )
