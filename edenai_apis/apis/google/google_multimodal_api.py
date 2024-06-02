from typing import Dict, List, Union, Generator, Optional
import json
import requests
import base64
from edenai_apis.features.multimodal.chat import (
    ChatDataClass,
    StreamChat,
    ChatMessageDataClass,
    ChatStreamResponse,
)
from edenai_apis.features.multimodal.multimodal_interface import MultimodalInterface
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.google.google_helpers import get_access_token
from edenai_apis.utils.exception import ProviderException


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
                "text": string,
                "inlineData": {
                    "mimeType": string,
                    "data": string
                },
                "fileData": {
                    "mimeType": string,
                    "fileUri": string
                }
                }
            ]
            }
        ]

        """
        transformed_messages = []
        for message in messages:
            role = message["role"].upper()
            if role == "USER":
                parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append({"text": content["content"]["text"]})
                    elif content["type"] == "media_url":
                        media_url = content["content"]["media_url"]
                        media_type = content["content"]["media_type"]
                        response = requests.get(media_url)

                        data = base64.b64encode(response.content).decode("utf-8")
                        parts.append(
                            {"inlineData": {"data": data, "mimeType": media_type}}
                        )
                    elif content["type"] == "media_base64":
                        media_base64 = content["content"]["media_base64"]
                        parts.append(
                            {
                                "inlineData": {
                                    "data": media_base64,
                                    "mimeType": content["content"]["media_type"],
                                }
                            }
                        )
                transformed_messages.append({"role": role, "parts": parts})
            elif role == "ASSISTANT":
                transformed_messages.append(
                    {
                        "role": role,
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
    def __calculate_usage_tokens(original_response: List[dict]) -> Dict:
        """
        Calculates the token usage from the original response.

        This function extracts token usage information from the provided response.
        It assumes that usageMetadata appears only in the last element of the object.

        """
        # The object usageMetadata appear only in the last element of the obj
        usage = {
            "prompt_tokens": original_response[-1]["usageMetadata"]["totalTokenCount"],
            "completion_tokens": original_response[-1]["usageMetadata"][
                "totalTokenCount"
            ],
            "total_tokens": original_response[-1]["usageMetadata"]["totalTokenCount"],
        }
        return {"usage": usage, "data": original_response}

    @staticmethod
    def __retrieve_generated_text(original_response: List[dict]) -> str:
        """
        Builds the generated text from Google Gemini response.

        This function constructs the generated text from the provided response.
        It iterates through the list of dictionaries and retrieves the text parts
        """
        answer = ""
        for resp in original_response:
            answer += (
                resp["candidates"][0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        return answer

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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        token = get_access_token(self.location)
        base_url = "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:streamGenerateContent"
        url = base_url.format(
            location="us-central1",
            project_id=self.project_id,
            model=model,
        )
        formatted_messages = self.__format_google_messages(messages=messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
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
            # "systemInstruction": chatbot_global_action,
        }
        if stream is False:
            response = requests.post(url, json=payload, headers=headers)
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
            generated_text = self.__retrieve_generated_text(
                original_response=original_response
            )
            original_response = self.__calculate_usage_tokens(
                original_response=original_response
            )
            standardized_response = ChatDataClass.generate_standardized_response(
                generated_text=generated_text,
                messages=messages,
            )

            return ResponseType[ChatDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )
        else:
            url += "?alt=sse"
            response = requests.post(url, json=payload, headers=headers, stream=True)
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
