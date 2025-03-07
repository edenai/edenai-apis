import json
from typing import Any, Dict, Optional

import requests

from edenai_apis.apis.amazon.helpers import check_webhook_result
from edenai_apis.features import ImageInterface, ProviderInterface, VideoInterface
from edenai_apis.features.image.deepfake_detection.deepfake_detection_dataclass import (
    DeepfakeDetectionDataClass as ImageDeepfakeDetectionDataclass,
)
from edenai_apis.features.video.deepfake_detection_async.deepfake_detection_async_dataclass import (
    DeepfakeDetectionAsyncDataClass as VideoDeepfakeDetectionAsyncDataclass,
)
from edenai_apis.features.video.deepfake_detection_async.deepfake_detection_async_dataclass import (
    DetailPerFrame,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.parsing import extract
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)


class SightEngineApi(ProviderInterface, ImageInterface, VideoInterface):
    provider_name = "sightengine"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY,
            provider_name=self.provider_name,
            api_keys=api_keys or {},
        )
        self.api_url = "https://api.sightengine.com/1.0"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {self.api_settings["api_key"]}',
        }
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings["webhook_token"]
        self.webhook_url = f"https://webhook.site/{self.webhook_token}"

    def image__deepfake_detection(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[ImageDeepfakeDetectionDataclass]:
        if not file_url and not file:
            raise ProviderException("file or file_url required")

        payload = {
            "url": file_url,
            "models": "deepfake",
            "api_user": self.api_settings["api_user"],
            "api_secret": self.api_settings["api_key"],
        }

        params = {
            "params": payload,
            "timeout": 30,
            "url": f"{self.api_url}/check.json",
            "method": "GET",
        }

        if not file_url:
            files = {"media": open(file, "rb")}
            payload.pop("url", None)
            params.pop("params", None)
            params["data"] = payload
            params["files"] = files
            params["method"] = "POST"

        try:
            response = requests.request(**params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ProviderException(f"Request failed: {str(e)}")

        original_response = response.json()
        score = extract(original_response, ["type", "deepfake"], None)
        if score is None:
            raise ProviderException("Deepfake score not found in response.")
        prediction = ImageDeepfakeDetectionDataclass.set_label_based_on_score(score)

        standardized_response = ImageDeepfakeDetectionDataclass(
            deepfake_score=score,
            prediction=prediction,
        )

        return ResponseType[ImageDeepfakeDetectionDataclass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def video__deepfake_detection_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        if not file_url and not file:
            raise ProviderException("file or file_url required")

        payload = {
            "models": "deepfake",
            "api_user": self.api_settings["api_user"],
            "api_secret": self.api_settings["api_key"],
            "callback_url": self.webhook_url,
        }

        method = "POST" if file else "GET"
        url = f"{self.api_url}/video/check.json"

        try:
            if file:
                with open(file, "rb") as video_file:
                    files = {"media": video_file}
                    response = requests.request(
                        method=method,
                        url=url,
                        files=files,
                        data=payload,
                        timeout=30,
                    )
            else:
                payload["stream_url"] = file_url
                response = requests.request(
                    method=method,
                    url=url,
                    params=payload,
                    timeout=30,
                )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ProviderException(f"Request failed: {str(e)}")

        original_response = response.json()
        media_id = original_response.get("media", {}).get("id")

        if not media_id:
            raise ProviderException("Media ID not found in response.")

        requests.post(
            self.webhook_url,
            json={"media_id": media_id},
            headers={"content-type": "application/json"},
        )

        return AsyncLaunchJobResponseType(provider_job_id=media_id)

    def video__deepfake_detection_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[VideoDeepfakeDetectionAsyncDataclass]:
        wehbook_result, response_status = check_webhook_result(
            provider_job_id, self.webhook_settings
        )

        if response_status != 200:
            raise ProviderException(wehbook_result, code=response_status)

        result_object = (
            next(
                filter(
                    lambda response: provider_job_id in response["content"],
                    wehbook_result,
                ),
                None,
            )
            if wehbook_result
            else None
        )

        if not result_object or not result_object.get("content"):
            return AsyncPendingResponseType[VideoDeepfakeDetectionAsyncDataclass](
                provider_job_id=provider_job_id
            )

        try:
            original_response = json.loads(result_object["content"])
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")

        if original_response is None:
            return AsyncPendingResponseType[VideoDeepfakeDetectionAsyncDataclass](
                provider_job_id=provider_job_id
            )

        score = extract(
            original_response, ["data", "frames", 0, "type", "deepfake"], None
        )
        if score is None:
            raise ProviderException("Deepfake score not found in response.")

        prediction = VideoDeepfakeDetectionAsyncDataclass.set_label_based_on_score(
            score
        )

        standardized_response = VideoDeepfakeDetectionAsyncDataclass(
            average_score=score,
            prediction=prediction,
            details_per_frame=[
                DetailPerFrame(
                    position=(frame.get("info", {}) or {}).get("position"),
                    score=(frame.get("type", {}) or {}).get("deepfake"),
                    prediction=VideoDeepfakeDetectionAsyncDataclass.set_label_based_on_score(
                        (frame.get("type", {}) or {}).get("deepfake")
                    ),
                )
                for frame in extract(original_response, ["data", "frames"], [])
            ],
        )

        return AsyncResponseType[VideoDeepfakeDetectionAsyncDataclass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )
