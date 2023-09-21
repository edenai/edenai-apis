from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass, AiDetectionItem,
)
from edenai_apis.utils.types import ResponseType
from typing import Dict, Sequence
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
import requests
import json
from edenai_apis.utils.exception import ProviderException

TOKEN_BEARER = ('Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
                '.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRkZ3NzdXRyaHpya2xsc3RnbGRiIiwicm9sZS'
                'I6ImFub24iLCJpYXQiOjE2ODY2ODc5MjM'
                'sImV4cCI6MjAwMjI2MzkyM30.bwSe1TrFMhcosgqFSlGIhMIv9fxohzLG0eyBEs7wUo8')


class WinstonaiApi(ProviderInterface, TextInterface):
    provider_name = "winstonai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.bearer_token = TOKEN_BEARER
        self.api_url = 'https://api.gowinston.ai/functions/v1'

    def text__ai_detection(self, text: str) -> ResponseType[AiDetectionDataClass]:
        payload = json.dumps({
            "api_key": self.api_key,
            "text": text,
            "sentences": True
        })

        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.bearer_token
        }

        response = requests.request(
            "POST", '{}/predict'.format(self.api_url), headers=headers, data=payload)

        if response.status_code != 200:
            raise ProviderException(response.json())

        original_response = response.json()
        score = original_response.get('score')
        sentences = original_response.get('sentences')

        if score is None or sentences is None:
            raise ProviderException(response.json())

        items: Sequence[AiDetectionItem] = [AiDetectionItem(text=sentence['text'], ai_score=sentence['score'],
                                                            prediction="") for sentence in sentences]

        standardized_response = AiDetectionDataClass(
            ai_score=score,
            items=items
        )

        return ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
