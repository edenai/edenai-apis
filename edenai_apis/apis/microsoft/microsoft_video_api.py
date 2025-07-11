import base64
import json
from io import BytesIO
from typing import Optional

import requests

from edenai_apis.features.video.generation_async.generation_async_dataclass import (
    GenerationAsyncDataClass,
)
from edenai_apis.features.video.video_interface import VideoInterface
from edenai_apis.utils.exception import (
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
)


class MicrosoftVideoApi(VideoInterface):
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
        base_url = self.azure_ai_credentials.get("azure_api_sora")
        api_key = self.azure_ai_credentials.get("azure_api_key")
        url = f"{base_url}openai/v1/video/generations/jobs?api-version=preview"
        try:
            height, width = dimension.split("x")
        except (ValueError, AttributeError) as exc:
            raise ProviderException(
                message=f"Invalid dimension format: {dimension}. Expected format: 'heightxwidth' (e.g., '1280x720')",
                code=400,
            ) from exc
        payload = {
            "model": model,
            "prompt": text,
            "height": height,
            "width": width,
            "n_seconds": duration,
        }
        response = requests.post(
            url=url, json=payload, headers={"Authorization": api_key}
        )
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        if response.status_code > 201:
            raise ProviderException(
                message=original_response["error"]["message"],
                code=response.status_code,
            )

        provider_job_id = original_response["id"]
        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__generation_async__get_job_result(
        self, provider_job_id: str
    ) -> GenerationAsyncDataClass:
        base_url = self.azure_ai_credentials.get("azure_api_sora")
        api_key = self.azure_ai_credentials.get("azure_api_key")
        url = f"{base_url}openai/v1/video/generations/jobs/{provider_job_id}?api-version=preview"

        response = requests.get(url, headers={"Authorization": api_key})
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        if "error" in original_response:
            raise ProviderException(
                message=original_response["error"]["message"],
                code=original_response["error"]["code"],
            )
        if original_response.get("status") == "succeeded":
            generations = original_response.get("generations", [])
            generation_id = generations[0].get("id")
            video_url = f"{base_url}openai/v1/video/generations/{generation_id}/content/video?api-version=preview"
            video_response = requests.get(video_url, headers={"Authorization": api_key})
            base64_encoded_string = base64.b64encode(video_response.content).decode(
                "utf-8"
            )
            resource_url = upload_file_bytes_to_s3(
                BytesIO(video_response.content), ".mp4", USER_PROCESS
            )
            standardized_response = GenerationAsyncDataClass(
                video=base64_encoded_string, video_resource_url=resource_url
            )
            return AsyncResponseType[GenerationAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType(provider_job_id=provider_job_id)
