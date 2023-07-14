from typing import Dict
import requests
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass,
    AiDetectionItem,
)
from edenai_apis.features import TextInterface
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from collections import defaultdict


class OriginalityaiApi(ProviderInterface, TextInterface):
    provider_name: str = "originalityai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.originality.ai/api/v1"

    def text__ai_detection(self, text: str) -> ResponseType[AiDetectionDataClass]:
        url = f"{self.base_url}/scan/ai"
        payload = {
            "title": "optional title",
            "content": text,
        }
        headers = {
            "content-type": "application/json",
            "X-OAI-API-KEY": self.api_key,
        }
        response = requests.post(url=url, headers=headers, json=payload)

        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(original_response.get("error"))

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

        result = ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
