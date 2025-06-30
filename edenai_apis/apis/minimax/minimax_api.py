import json
import base64
from io import BytesIO

from typing import Dict, List, Literal, Optional, Type, Union
import requests
import httpx
from openai import BaseModel, OpenAI

from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.llmengine.types.response_types import ResponseModel
from edenai_apis.features import (
    ImageInterface,
    ProviderInterface,
    VideoInterface,
    AudioInterface,
)
from edenai_apis.features.llm.chat.chat_dataclass import (
    StreamChat as StreamChatCompletion,
)
from edenai_apis.features.video import GenerationAsyncDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.features.llm.llm_interface import LlmInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3


class MinimaxApi(
    ProviderInterface, ImageInterface, LlmInterface, VideoInterface, AudioInterface
):
    provider_name = "minimax"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.base_url = "https://api.minimax.io/v1"
        self.api_key = self.api_settings.get("api_key")
        self.group_id = self.api_settings.get("group_id")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def video__generation_async__launch_job(
        self,
        text: str,
        duration: Optional[int] = 6,
        fps: Optional[int] = 24,
        dimension: Optional[str] = "1280x720",
        resolution: Optional[str] = "720p",
        seed: Optional[float] = 12,
        file: Optional[str] = None,
        file_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        payload = {
            "prompt": text,
            "model": model,
            "duration": duration,
            # "resolution": dimension,
        }
        if file:
            with open(file, "rb") as file_:
                file_content = file_.read()
                input_image_base64 = base64.b64encode(file_content).decode("utf-8")
                payload["first_frame_images"] = input_image_base64
        response = requests.post(
            url=f"{self.base_url}/video_generation",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal server error", code=500) from exc
        base_msg = original_response.get("base_resp")
        if base_msg["status_msg"] != "success":
            raise ProviderException(
                message=base_msg["status_msg"], code=base_msg["status_code"]
            )
        provider_job_id = original_response.get("task_id")
        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__generation_async__get_job_result(
        self, provider_job_id: str
    ) -> GenerationAsyncDataClass:
        response = requests.get(
            url=f"{self.base_url}/query/video_generation?task_id={provider_job_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal server error", code=500) from exc
        base_msg = original_response.get("base_resp")
        if base_msg["status_msg"] != "success":
            raise ProviderException(
                message=base_msg["status_msg"], code=base_msg["status_code"]
            )
        if original_response["status"] == "Success":
            file_id = original_response["file_id"]
            file_response = requests.get(
                url=f"{self.base_url}/files/retrieve?GroupId={self.group_id}&file_id={file_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            file_response_data = file_response.json()
            base_msg = file_response_data.get("base_resp")
            if base_msg["status_msg"] != "success":
                raise ProviderException(
                    message=base_msg["status_msg"], code=base_msg["status_code"]
                )
            return AsyncResponseType(
                original_response=original_response,
                standardized_response=GenerationAsyncDataClass(
                    video="",
                    video_resource_url=file_response_data["file"]["download_url"],
                ),
                provider_job_id=provider_job_id,
            )

        if original_response["status"] == "Fail":
            failure_message = original_response["base_resp"]["status_msg"]
            raise ProviderException(failure_message)

        else:
            return AsyncPendingResponseType(provider_job_id=provider_job_id)

    def llm__chat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        modalities: Optional[List[Literal["text", "audio", "image"]]] = None,
        audio: Optional[Dict] = None,
        # openai v1.0+ new params
        response_format: Optional[
            Union[dict, Type[BaseModel]]
        ] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This should be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ):
        completion_params = {"messages": messages, "model": model}
        if response_format is not None:
            completion_params["response_format"] = response_format
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        if temperature is not None:
            completion_params["temperature"] = temperature
        if tools is not None:
            completion_params["tools"] = tools
        if top_p is not None:
            completion_params["top_p"] = top_p
        if stream is not None:
            completion_params["stream"] = stream
        try:
            response = self.client.chat.completions.create(**completion_params)
            if stream:

                def generate_chunks():
                    for chunk in response:
                        if chunk is not None:
                            yield chunk.to_dict()
                            # yield ModelResponseStream.model(data)

                return StreamChatCompletion(stream=generate_chunks())
            else:
                response = response.to_dict()
                base_resp = response["base_resp"]
                if base_resp["status_msg"] != "":
                    raise ProviderException(
                        message=base_resp["status_msg"],
                        code=base_resp["status_code"],
                    )
                response_model = ResponseModel.model_validate(response)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        return response_model

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        height, width = resolution.split("x")
        payload = {
            "prompt": text,
            "model": model,
            "width": int(width),
            "height": int(height),
            "n": num_images,
            "response_format": "base64",
        }
        response = requests.post(
            url=f"{self.base_url}/image_generation",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal server error", code=500) from exc
        base_msg = original_response.get("base_resp")
        if base_msg["status_msg"] != "success":
            raise ProviderException(
                message=base_msg["status_msg"], code=base_msg["status_code"]
            )
        generations = []
        for generated_image in original_response["data"]["image_base64"]:
            image_data = generated_image.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(
                image_content, ".png", "users_process"
            )
            generations.append(
                GeneratedImageDataClass(
                    image=generated_image, image_resource_url=resource_url
                )
            )
        standardized_response = GenerationDataClass(items=generations)
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
