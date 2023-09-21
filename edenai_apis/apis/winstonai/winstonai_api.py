from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass,
)
from edenai_apis.utils.types import ResponseType
from typing import Dict
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
import requests
import json
from edenai_apis.utils.exception import ProviderException

BEARER_TOKEN = ('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
                '.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRkZ3NzdXRyaHpya2xsc3RnbGRiIiwic'
                'm9sZSI6ImFub24iLCJpYXQiOjE2O'
                'DY2ODc5MjMsImV4cCI6MjAwMjI2MzkyM30.bwSe1TrFMh'
                'cosgqFSlGIhMIv9fxohzLG0eyBEs7wUo8')


class WinstonaiApi(ProviderInterface, TextInterface):
    provider_name = "winstonai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.bearer_token = BEARER_TOKEN
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

        response = requests.request("POST", '{}/predict'.format(self.api_url), headers=headers, data=payload)

        if response.status_code != 200:
            raise ProviderException(code=response.status_code)

        original_response = response.json()

        standardized_response = AiDetectionDataClass(
            ai_score=original_response.get('score')
        )

        result = ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
