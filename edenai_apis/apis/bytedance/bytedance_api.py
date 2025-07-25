import base64
import json
from io import BytesIO
from typing import Dict, Literal, Optional, Any, List
import mimetypes

import requests

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
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from edenai_apis.llmengine.utils.moderation import moderate


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
