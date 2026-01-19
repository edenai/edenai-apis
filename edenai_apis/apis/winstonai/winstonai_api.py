import json
from http import HTTPStatus
from typing import Any, Dict, Optional, Sequence
from uuid import uuid4

import requests

from edenai_apis.apis.winstonai.config import WINSTON_AI_API_URL
from edenai_apis.utils.http_client import DEFAULT_TIMEOUT, async_client
from edenai_apis.features import ImageInterface, ProviderInterface, TextInterface
from edenai_apis.features.image.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass as ImageAiDetectionDataclass,
)
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass,
    AiDetectionItem,
)
from edenai_apis.features.text.plagia_detection.plagia_detection_dataclass import (
    PlagiaDetectionCandidate,
    PlagiaDetectionDataClass,
    PlagiaDetectionItem,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import aupload_file_bytes_to_s3, upload_file_to_s3


class WinstonaiApi(ProviderInterface, TextInterface, ImageInterface):
    provider_name = "winstonai"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        if api_keys is not None:
            self.api_settings = api_keys
        else:
            self.api_settings = load_provider(
                ProviderDataEnum.KEY,
                provider_name=self.provider_name,
                api_keys=api_keys or {},
            )
        self.api_url = WINSTON_AI_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {self.api_settings["api_key"]}',
        }

    def image__ai_detection(
        self, file: Optional[str] = None, file_url: Optional[str] = None, **kwargs
    ) -> ResponseType[ImageAiDetectionDataclass]:
        if not file_url and not file:
            raise ProviderException("file or file_url required")

        payload = json.dumps({"url": file_url or upload_file_to_s3(file, file)})

        response = requests.request(
            "POST",
            f"{self.api_url}/image-detection",
            headers=self.headers,
            data=payload,
        )

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()

        score = 1 - original_response.get("score") / 100
        prediction = ImageAiDetectionDataclass.set_label_based_on_score(score)
        if score is None:
            raise ProviderException(response.json())

        standardized_response = ImageAiDetectionDataclass(
            ai_score=score,
            prediction=prediction,
        )

        return ResponseType[ImageAiDetectionDataclass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    async def image__aai_detection(
        self, file: Optional[str] = None, file_url: Optional[str] = None, **kwargs
    ) -> ResponseType[ImageAiDetectionDataclass]:
        if not file_url and not file:
            raise ProviderException("file or file_url required")

        if file and not file_url:
            from io import BytesIO

            with open(file, "rb") as f:
                file_bytes = BytesIO(f.read())
            url = await aupload_file_bytes_to_s3(file_bytes, file)
        else:
            url = file_url

        payload = json.dumps({"url": url})

        async with async_client(DEFAULT_TIMEOUT) as client:
            response = await client.request(
                "POST",
                f"{self.api_url}/image-detection",
                headers=self.headers,
                data=payload,
            )

            if response.status_code != 200:
                raise ProviderException(response.json(), code=response.status_code)

            original_response = response.json()

            score = 1 - original_response.get("score") / 100
            prediction = ImageAiDetectionDataclass.set_label_based_on_score(score)
            if score is None:
                raise ProviderException(response.json())

            standardized_response = ImageAiDetectionDataclass(
                ai_score=score,
                prediction=prediction,
            )

            return ResponseType[ImageAiDetectionDataclass](
                original_response=original_response,
                standardized_response=standardized_response,
            )

    def text__ai_detection(
        self, text: str, provider_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ResponseType[AiDetectionDataClass]:
        if provider_params is None:
            provider_params = {}
        payload = json.dumps(
            {
                "text": text,
                "sentences": True,
                "language": provider_params.get("language", "en"),
                "version": provider_params.get("version", "2.0"),
            }
        )

        response = requests.request(
            "POST", f"{self.api_url}/predict", headers=self.headers, data=payload
        )

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal Server Error")

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        score = original_response.get("score") / 100
        sentences = original_response.get("sentences")

        if score is None or sentences is None:
            raise ProviderException(response.json())

        items: Sequence[AiDetectionItem] = [
            AiDetectionItem(
                text=sentence["text"],
                ai_score=1 - (sentence["score"] / 100),
                prediction=AiDetectionItem.set_label_based_on_score(
                    1 - (sentence["score"] / 100)
                ),
            )
            for sentence in sentences
        ]

        standardized_response = AiDetectionDataClass(ai_score=1 - score, items=items)

        return ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    async def text__aai_detection(
        self, text: str, provider_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ResponseType[AiDetectionDataClass]:
        if provider_params is None:
            provider_params = {}
        payload = {
            "text": text,
            "sentences": True,
            "language": provider_params.get("language", "en"),
            "version": provider_params.get("version", "2.0"),
        }

        async with async_client(DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self.api_url}/predict", headers=self.headers, json=payload
            )

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal Server Error")

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        score = original_response.get("score") / 100
        sentences = original_response.get("sentences")

        if score is None or sentences is None:
            raise ProviderException(response.json())

        items: Sequence[AiDetectionItem] = [
            AiDetectionItem(
                text=sentence["text"],
                ai_score=1 - (sentence["score"] / 100),
                prediction=AiDetectionItem.set_label_based_on_score(
                    1 - (sentence["score"] / 100)
                ),
            )
            for sentence in sentences
        ]

        standardized_response = AiDetectionDataClass(ai_score=1 - score, items=items)

        return ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__plagia_detection(
        self,
        text: str,
        title: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[PlagiaDetectionDataClass]:
        if provider_params is None:
            provider_params = {}
        payload = json.dumps(
            {
                "text": text,
                "language": provider_params.get("language", "en"),
                "version": provider_params.get("version", "2.0"),
            }
        )

        response = requests.request(
            "POST", f"{self.api_url}/plagiarism", headers=self.headers, data=payload
        )

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        results = original_response.get("results")

        if results is None:
            raise ProviderException(response.json())

        standardized_response = PlagiaDetectionDataClass(
            plagia_score=original_response["score"],
            items=[
                PlagiaDetectionItem(
                    text=result["title"],
                    candidates=[
                        PlagiaDetectionCandidate(
                            url=result["url"],
                            plagia_score=1,
                            prediction="plagiarized",
                            plagiarized_text=excerpt,
                        )
                        for excerpt in result["excerpts"]
                    ],
                )
                for result in results
            ],
        )

        return ResponseType[PlagiaDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    async def text__aplagia_detection(
        self,
        text: str,
        title: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[PlagiaDetectionDataClass]:
        if provider_params is None:
            provider_params = {}
        payload = json.dumps(
            {
                "text": text,
                "language": provider_params.get("language", "en"),
                "version": provider_params.get("version", "2.0"),
            }
        )
        async with async_client(DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self.api_url}/plagiarism", headers=self.headers, data=payload
            )

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal Server Error")

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        results = original_response.get("results")

        if results is None:
            raise ProviderException(response.json())

        standardized_response = PlagiaDetectionDataClass(
            plagia_score=original_response["score"],
            items=[
                PlagiaDetectionItem(
                    text=result["title"],
                    candidates=[
                        PlagiaDetectionCandidate(
                            url=result["url"],
                            plagia_score=1,
                            prediction="plagiarized",
                            plagiarized_text=excerpt,
                        )
                        for excerpt in result["excerpts"]
                    ],
                )
                for result in results
            ],
        )

        return ResponseType[PlagiaDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
