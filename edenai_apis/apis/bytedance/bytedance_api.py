import asyncio
import base64
import json
from io import BytesIO
from openai import BaseModel

from typing import Dict, List, Type, Union, Optional, Literal, Any
import httpx
import requests

from edenai_apis.features.llm.chat.chat_dataclass import ChatDataClass
from edenai_apis.features import ProviderInterface, ImageInterface, VideoInterface
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.video.generation_async import GenerationAsyncDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    ResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    aupload_file_bytes_to_s3,
    upload_file_bytes_to_s3,
)
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.llmengine.utils.moderation import async_moderate, moderate


class BytedanceApi(ProviderInterface, ImageInterface, VideoInterface):
    provider_name = "bytedance"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.llm_client = LLMEngine(
            provider_name="openai",
            provider_config={
                "api_key": self.api_key,
            },
        )

    @moderate
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        payload = {
            "model": model,
            "prompt": text,
            "size": resolution,
            "response_format": "b64_json",
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        # Handle error
        if response.status_code != 200:
            raise ProviderException(
                message=original_response.get("error", {}).get("message"),
                code=response.status_code,
            )

        generations: List[GeneratedImageDataClass] = []
        for generated_image in original_response.get("data"):
            image_b64 = generated_image.get("b64_json")

            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                GeneratedImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(items=generations),
        )

    @async_moderate
    async def image__ageneration(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        payload = {
            "model": model,
            "prompt": text,
            "size": resolution,
            "response_format": "b64_json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload, headers=self.headers)
            try:
                original_response = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException("Internal Server Error", code=500) from exc

            # Handle error
            if response.status_code != 200:
                raise ProviderException(
                    message=original_response.get("error", {}).get("message"),
                    code=response.status_code,
                )

            # Process images and upload to S3 concurrently
            # Bytedance sends a dict with the b64 image in a dict["b64_json"]
            async def process_and_upload_image(image_b64: dict):
                image = image_b64.get("b64_json")

                # Decode base64 in thread pool (CPU-bound operation)
                def decode_image():
                    base64_bytes = image.encode()
                    return BytesIO(base64.b64decode(base64_bytes))

                image_bytes = await asyncio.to_thread(decode_image)

                # Upload to S3 asynchronously
                resource_url = await aupload_file_bytes_to_s3(
                    image_bytes, ".png", USER_PROCESS
                )

                return GeneratedImageDataClass(
                    image=image, image_resource_url=resource_url
                )

            # Process and upload all images concurrently
            generated_images = await asyncio.gather(
                *[
                    process_and_upload_image(image)
                    for image in original_response.get("data")
                ]
            )

            return ResponseType[GenerationDataClass](
                original_response=original_response,
                standardized_response=GenerationDataClass(items=list(generated_images)),
            )

    def video__generation_async__launch_job(
        self,
        text: str,
        duration: Optional[int] = 6,
        fps: Optional[int] = 24,
        dimension: Optional[str] = "1280x720",
        seed: Optional[float] = 12,
        file: Optional[str] = None,
        file_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        url = (
            "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
        )
        content = [
            {"type": "text", "text": text},
        ]
        # if file:
        #     with open(file, "rb") as fstream:
        #         file_content = fstream.read()
        #         file_b64 = base64.b64encode(file_content).decode("utf-8")
        #     mime_type = mimetypes.guess_type(file)[0]
        #     image_data = f"data:{mime_type};base64,{file_b64}"
        #     content.append({"type": "image_url", "image_url": {"url": image_data}})
        payload = {
            "model": model,
            "content": content,
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        # Handle error
        if response.status_code != 200:
            raise ProviderException(
                message=original_response.get("error", {}).get("message"),
                code=response.status_code,
            )
        provider_job_id = original_response.get("id")
        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__generation_async__get_job_result(
        self, provider_job_id: str
    ) -> GenerationAsyncDataClass:
        url = f"https://ark.ap-southeast.volces.com/api/v3/contents/generations/tasks/{provider_job_id}"
        try:
            response = requests.get(url, headers=self.headers)
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["error"]["message"], code=response.status_code
            )
        if original_response["status"] == "cancelled":
            failure_message = original_response["error"]
            raise ProviderException(failure_message)
        if original_response["status"] != "succeeded":
            return AsyncPendingResponseType(provider_job_id=provider_job_id)
        video_uri = original_response["content"]["video_url"]
        video_response = requests.get(video_uri)
        base64_encoded_string = base64.b64encode(video_response.content).decode("utf-8")
        resource_url = upload_file_bytes_to_s3(
            BytesIO(video_response.content), ".mp4", USER_PROCESS
        )

        return AsyncResponseType(
            original_response=original_response,
            standardized_response=GenerationAsyncDataClass(
                video=base64_encoded_string, video_resource_url=resource_url
            ),
            provider_job_id=provider_job_id,
        )

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
    ) -> ChatDataClass:
        response = self.llm_client.completion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_version=api_version,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response

    async def llm__achat(
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
    ) -> ChatDataClass:
        response = await self.llm_client.acompletion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_version=api_version,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response
