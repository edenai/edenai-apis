import base64
from typing import Dict, Generator, List, Literal, Optional, Union, overload

import requests

from edenai_apis.features import TextInterface, ImageInterface
from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text import (
    ChatDataClass,
    ChatMessageDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatStreamResponse
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from .config import get_model_id, get_model_id_image


class ReplicateApi(ProviderInterface, ImageInterface, TextInterface):
    provider_name = "replicate"

    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Token {api_settings['api_key']}",
        }
        self.base_url = "https://api.replicate.com/v1"
        
    def __get_stream_response(self, url: str) -> Generator:
        headers = {**self.headers, "Accept": "text/event-stream"}
        response = requests.get(url, headers=headers, stream=True)
        last_chunk = ""
        for chunk in response.iter_lines():
            if b"event: done" in chunk:
                response.close()
                break
            elif last_chunk == b"event: error" and chunk.startswith(b"data: "):
                yield ChatStreamResponse(text = "[ERROR]",
                                             blocked = True, 
                                             provider = self.provider_name)
            elif chunk.startswith(b"data: "):
                if last_chunk == b"data: " and chunk == b"data: ":
                    yield ChatStreamResponse(text = "\n", blocked = False, provider = self.provider_name)
                else:
                    yield ChatStreamResponse(text = chunk.decode("utf-8").replace("data: ", ""),
                                             blocked = False, 
                                             provider = self.provider_name)
            last_chunk = chunk

    @overload
    def __get_response(
        self, url: str, payload: dict, stream: Literal[True]
    ) -> Generator:
        ...

    @overload
    def __get_response(self, url: str, payload: dict, stream: Literal[False]) -> dict:
        ...

    def __get_response(
        self, url: str, payload: dict, stream: bool = False
    ) -> Union[Generator, dict]:
        # Launch job
        if stream:
            payload["stream"] = True
        launch_job_response = requests.post(url, headers=self.headers, json=payload)
        try:
            launch_job_response_dict = launch_job_response.json()
        except requests.JSONDecodeError:
            raise ProviderException(
                launch_job_response.text, code=launch_job_response.status_code
            )
        if launch_job_response.status_code != 201:
            raise ProviderException(
                launch_job_response_dict.get("detail"),
                code=launch_job_response.status_code,
            )

        if stream:
            return self.__get_stream_response(
                launch_job_response_dict["urls"]["stream"]
            )
        url_get_response = launch_job_response_dict["urls"]["get"]

        # Get job response
        response = requests.get(url_get_response, headers=self.headers)
        response_dict = response.json()
        if response.status_code != 200:
            raise ProviderException(
                response_dict.get("detail"), code=response.status_code
            )

        status = response_dict["status"]
        while status != "succeeded":
            response = requests.get(url_get_response, headers=self.headers)
            try:
                response_dict = response.json()
            except requests.JSONDecodeError:
                raise ProviderException(response.text, code=response.status_code)

            if response.status_code != 200:
                raise ProviderException(
                    response_dict.get("error", response_dict), code=response.status_code
                )

            status = response_dict["status"]

        return response_dict

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}/predictions"
        size = resolution.split("x")
        version = get_model_id_image[model]

        payload = {
            "input": {
                "prompt": text,
                "width": int(size[0]),
                "height": int(size[1]),
            },
            "version": version,
        }

        response_dict = ReplicateApi.__get_response(self, url, payload)
        image_url = response_dict.get("output")
        if isinstance(image_url, list):
            image_url = image_url[0]
        image_bytes = base64.b64encode(requests.get(image_url).content)

        return ResponseType[GenerationDataClass](
            original_response=response_dict,
            standardized_response=GenerationDataClass(
                items=[
                    GeneratedImageDataClass(
                        image=image_bytes, image_resource_url=image_url
                    )
                ]
            ),
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream: bool = False,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        # Construct the API URL
        url = f"{self.base_url}/predictions"

        # Get the model ID based on the provided model name
        model_id = get_model_id[model]

        # Build the prompt by formatting the previous history and current text
        prompt = ""
        if previous_history:
            for msg in previous_history:
                if msg["role"] == "user":
                    prompt += "\n[INST]" + msg["message"] + "[/INST]\n"
                else:
                    prompt += "\n" + msg["message"] + "\n"

        prompt += "\n[INST]" + text + "[/INST]\n"

        # Construct the payload for chat interaction
        payload = {
            "input": {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "min_new_tokens": -1,
            },
            "version": model_id,
        }

        # Include system prompt if provided
        if chatbot_global_action:
            payload["input"]["system_prompt"] = chatbot_global_action

        # Call the API and get the response dictionary
        response = self.__get_response(url, payload, stream=stream)

        if stream is False:
            # Extract generated text from the API response
            generated_text = "".join(response.get("output", [""]))

            # Build a list of ChatMessageDataClass objects for the conversation history
            message = [
                ChatMessageDataClass(role="user", message=text),
                ChatMessageDataClass(role="assistant", message=generated_text),
            ]

            # Build the standardized response
            standardized_response = ChatDataClass(
                generated_text=generated_text, message=message
            )

            return ResponseType[ChatDataClass](
                original_response=response,
                standardized_response=standardized_response,
            )
        else:
            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=response),
            )
