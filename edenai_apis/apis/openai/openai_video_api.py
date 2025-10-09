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
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncErrorResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
)


class OpenaiVideoApi(VideoInterface):

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
        url = "https://api.openai.com/v1/videos"
        payload = {
            "prompt": text,
            "model": model,
            "seconds": str(duration),
            "size": dimension,
        }
        response = requests.post(url=url, headers=self.headers, json=payload)
        try:
            provider_job_response = response.json()
        except json.JSONDecodeError as exp:
            raise ProviderException(
                message="Internal server error", code=response.status_code
            ) from exp

        if response.status_code != 200:
            raise ProviderException(
                message=provider_job_response["error"]["message"],
                code=response.status_code,
            )

        provider_job_id = provider_job_response["id"]

        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__generation_async__get_job_result(
        self,
        provider_job_id: str,
    ) -> AsyncBaseResponseType[GenerationAsyncDataClass]:
        get_video_url = f"https://api.openai.com/v1/videos/{provider_job_id}"
        response = requests.get(url=get_video_url, headers=self.headers)
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
        status = original_response["status"]
        if status == "completed":
            response = requests.get(
                url=f"{get_video_url}/content", headers=self.headers
            )
            file_content = response.content
            base64_encoded_string = base64.b64encode(file_content).decode("utf-8")
            resource_url = upload_file_bytes_to_s3(
                BytesIO(file_content), ".mp4", USER_PROCESS
            )
            standardized_response = GenerationAsyncDataClass(
                video=base64_encoded_string, video_resource_url=resource_url
            )
            return AsyncResponseType[GenerationAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )
        if status == "failed":
            return AsyncErrorResponseType(
                provider_job_id=provider_job_id,
                error=original_response["error"]["message"],
            )

        return AsyncPendingResponseType(provider_job_id=provider_job_id)
