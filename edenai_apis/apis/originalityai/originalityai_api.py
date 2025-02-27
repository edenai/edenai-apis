import json
from collections import defaultdict
from http import HTTPStatus
from typing import Any, Dict, Optional

import requests

from edenai_apis.features import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
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


class OriginalityaiApi(ProviderInterface, TextInterface):
    provider_name: str = "originalityai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.originality.ai/api/v1/scan"

    def text__plagia_detection(
        self,
        text: str,
        title: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[PlagiaDetectionDataClass]:
        url = f"{self.base_url}/plag"
        payload = {"content": text, "title": title}
        headers = {"content-type": "application/json", "X-OAI-API-KEY": self.api_key}

        response = requests.post(url, headers=headers, json=payload)

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message="Internal Server Error", code=response.status_code
            ) from exc

        if response.status_code != HTTPStatus.OK:
            raise ProviderException(
                original_response.get("message") or original_response,
                code=response.status_code,
            )

        total_score = float(
            int(original_response["total_text_score"].replace("%", "")) / 100
        )

        items = []
        for result in original_response.get("results", []) or []:
            text = result["phrase"]
            candidates = []
            for match in result.get("matches", []) or []:
                candidates.append(
                    PlagiaDetectionCandidate(
                        url=match["website"],
                        plagia_score=int(match["score"]) / 100,
                        plagiarized_text=match["pText"],
                    )
                )
            items.append(PlagiaDetectionItem(text=text, candidates=candidates))

        standardized_response = PlagiaDetectionDataClass(
            plagia_score=total_score, items=items
        )

        # remove credits information from original response
        original_response.pop("credits_used")
        original_response.pop("credits")

        result = ResponseType[PlagiaDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result

    def text__ai_detection(
        self, text: str, provider_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ResponseType[AiDetectionDataClass]:
        url = f"{self.base_url}/ai"
        payload = {
            "title": "optional title",
            "content": text,
        }
        headers = {
            "content-type": "application/json",
            "X-OAI-API-KEY": self.api_key,
        }
        response = requests.post(url=url, headers=headers, json=payload)

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message="Internal Server Error", code=response.status_code
            ) from exc

        if response.status_code != HTTPStatus.OK:
            raise ProviderException(
                original_response.get("message") or original_response,
                code=response.status_code,
            )

        default_dict = defaultdict(lambda: None)
        items = []
        for block in original_response.get("blocks"):
            ai_score = block.get("result", default_dict).get("fake")
            items.append(
                AiDetectionItem(
                    text=block.get("text"),
                    prediction=AiDetectionItem.set_label_based_on_score(ai_score),
                    ai_score=ai_score,
                )
            )

        standardized_response = AiDetectionDataClass(
            ai_score=original_response.get("score", defaultdict).get("ai"), items=items
        )

        # remove credits information from original response
        original_response.pop("credits_used")
        original_response.pop("credits")

        result = ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
