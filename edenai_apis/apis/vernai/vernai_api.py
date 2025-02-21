from typing import Dict, Sequence

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    EmotionDetectionDataClass,
    EmotionItem,
    EmotionEnum,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class VernaiApi(ProviderInterface, TextInterface):
    provider_name = "vernai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url_emotion_detection = "https://vernapi.com/analyze"

    def text__emotion_detection(
        self, text: str, **kwargs
    ) -> ResponseType[EmotionDetectionDataClass]:
        response = requests.post(
            url=self.url_emotion_detection,
            headers={"Authorization": f"{self.api_key}"},
            data={"text": text},
        )
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        original_response = response.json()
        items: Sequence[EmotionItem] = []
        for entity in original_response.get("scores", []):
            items.append(
                EmotionItem(
                    emotion=EmotionEnum.from_str(entity.get("name", "")),
                    emotion_score=entity.get("value", 0),
                )
            )
        return ResponseType[EmotionDetectionDataClass](
            original_response=original_response,
            standardized_response=EmotionDetectionDataClass(items=items, text=text),
        )
