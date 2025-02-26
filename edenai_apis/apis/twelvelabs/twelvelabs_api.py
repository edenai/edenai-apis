import random
from typing import Dict

import requests

from edenai_apis.apis.twelvelabs.helpers import (
    convert_json_to_logo_dataclass,
    convert_json_to_text_dataclass,
)
from edenai_apis.features import ProviderInterface, VideoInterface
from edenai_apis.features.video.logo_detection_async.logo_detection_async_dataclass import (
    LogoDetectionAsyncDataClass,
)
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    TextDetectionAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)


class TwelveLabsApi(ProviderInterface, VideoInterface):
    provider_name = "twelvelabs"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )

        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.twelvelabs.io/v1.1"
        self.headers = {"x-api-key": self.api_key}

    def video__logo_detection_async__launch_job(
        self, file: str, file_url: str = "", language: str = "en", **kwargs
    ) -> AsyncLaunchJobResponseType:

        index_url = f"{self.base_url}/indexes"
        task_url = f"{self.base_url}/tasks"

        index_data_config = {
            "engine_id": "marengo2.5",
            "index_options": ["logo"],
            "index_name": str(random.randint(0, 10000000)),
        }

        response = requests.post(
            index_url, headers=self.headers, json=index_data_config
        )

        if response.status_code != 201:
            raise ProviderException(message=response.text, code=response.status_code)

        index_id = response.json().get("_id")

        if not file_url:
            file_name = str(file).split("/")[-1]
            file_stream = open(file, "rb")
            file_param = [
                ("video_file", (file_name, file_stream, "application/octet-stream"))
            ]

        else:
            file_param = [("video_url", file_url)]

        video_data_config = {
            "index_id": index_id,
            "language": language,
            "disable_video_stream": "false",
        }

        response = requests.post(
            task_url, headers=self.headers, data=video_data_config, files=file_param
        )
        if file_stream is not None:
            file_stream.close()

        if response.status_code != 201:
            raise ProviderException(message=response.text, code=response.status_code)

        task_id = response.json().get("_id")

        status_task_url = f"{self.base_url}/tasks/{task_id}"

        response = requests.get(status_task_url, headers=self.headers)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        response = response.json()

        video_id = response.get("video_id")

        provider_job_id = index_id + "_" + video_id + "_" + task_id

        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__logo_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[LogoDetectionAsyncDataClass]:

        try:
            index_id, video_id, task_id = provider_job_id.split("_")
        except ValueError:
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)

        task_url = f"{self.base_url}/indexes/{index_id}/videos/{video_id}/logo"
        status_task_url = f"{self.base_url}/tasks/{task_id}"

        response = requests.get(task_url, headers=self.headers)

        if response.status_code == 422:
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        original_response = response.json()

        if original_response.get("data") is None:
            response = requests.get(status_task_url, headers=self.headers)
            if response.status_code != 200:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )

            task_status = response.json().get("status")

            if task_status != "ready":
                return AsyncPendingResponseType[LogoDetectionAsyncDataClass](
                    status="pending",
                    provider_job_id=provider_job_id,
                )

        url = f"https://api.twelvelabs.io/v1.1/indexes/{index_id}"

        response = requests.delete(url, headers=self.headers)

        if response.status_code != 204:
            raise ProviderException(message=response.text, code=response.status_code)

        return AsyncResponseType[LogoDetectionAsyncDataClass](
            status="succeeded",
            original_response=original_response,
            standardized_response=convert_json_to_logo_dataclass(original_response),
            provider_job_id=provider_job_id,
        )

    def video__text_detection_async__launch_job(
        self, file: str, file_url: str = "", language: str = "en", **kwargs
    ) -> AsyncLaunchJobResponseType:

        index_url = f"{self.base_url}/indexes"
        task_url = f"{self.base_url}/tasks"

        index_data_config = {
            "engine_id": "marengo2.5",
            "index_options": ["text_in_video"],
            "index_name": str(random.randint(0, 10000000)),
        }

        # Create index
        response = requests.post(
            index_url, headers=self.headers, json=index_data_config
        )

        if response.status_code != 201:
            raise ProviderException(message=response.text, code=response.status_code)

        index_id = response.json().get("_id")

        if not file_url:
            file_name = str(file).split("/")[-1]
            file_stream = open(file, "rb")
            file_param = [
                ("video_file", (file_name, file_stream, "application/octet-stream"))
            ]

        else:
            file_param = [("video_url", file_url)]

        video_data_config = {
            "index_id": index_id,
            "language": language,
            "disable_video_stream": "false",
        }

        # Create video task
        response = requests.post(
            task_url, headers=self.headers, data=video_data_config, files=file_param
        )
        if file_stream is not None:
            file_stream.close()
        if response.status_code != 201:
            raise ProviderException(message=response.text, code=response.status_code)

        task_id = response.json().get("_id")

        status_task_url = f"{self.base_url}/tasks/{task_id}"

        response = requests.get(status_task_url, headers=self.headers)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        response = response.json()

        video_id = response.get("video_id")

        provider_job_id = index_id + "_" + video_id + "_" + task_id

        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)

    def video__text_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[TextDetectionAsyncDataClass]:
        try:
            index_id, video_id, task_id = provider_job_id.split("_")
        except ValueError:
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)

        task_url = f"{self.base_url}/indexes/{index_id}/videos/{video_id}/text-in-video"
        status_task_url = f"{self.base_url}/tasks/{task_id}"

        response = requests.get(task_url, headers=self.headers)

        if response.status_code == 422:
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        original_response = response.json()

        if original_response.get("data") is None:

            # check task status
            response = requests.get(status_task_url, headers=self.headers)
            if response.status_code != 200:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )

            task_status = response.json().get("status")

            if task_status != "ready":
                return AsyncPendingResponseType[TextDetectionAsyncDataClass](
                    status="pending",
                    provider_job_id=provider_job_id,
                )

        url = f"https://api.twelvelabs.io/v1.1/indexes/{index_id}"

        response = requests.delete(url, headers=self.headers)

        if response.status_code != 204:
            raise ProviderException(message=response.text, code=response.status_code)

        return AsyncResponseType[TextDetectionAsyncDataClass](
            status="succeeded",
            original_response=original_response,
            standardized_response=convert_json_to_text_dataclass(original_response),
            provider_job_id=provider_job_id,
        )
