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
from edenai_apis.apis.winstonai.config import TOKEN_BEARER, WINSTON_AI_API_URL
from edenai_apis.features.text.plagia_detection.plagia_detection_dataclass import PlagiaDetectionCandidate, PlagiaDetectionDataClass, PlagiaDetectionItem


class WinstonaiApi(ProviderInterface, TextInterface):
    provider_name = "winstonai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.bearer_token = TOKEN_BEARER
        self.api_url = WINSTON_AI_API_URL
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': self.bearer_token
        }

    def text__ai_detection(self, text: str) -> ResponseType[AiDetectionDataClass]:
        payload = json.dumps({
            "api_key": self.api_key,
            "text": text,
            "sentences": True
        })

        response = requests.request(
            "POST", f'{self.api_url}/predict', headers=self.headers, data=payload)

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        score = original_response.get('score')
        sentences = original_response.get('sentences')

        if score is None or sentences is None:
            raise ProviderException(response.json())

        items: Sequence[AiDetectionItem] = [AiDetectionItem(text=sentence['text'],
                                                            ai_score=sentence['score'],
                                                            prediction=AiDetectionItem.set_label_based_on_score(
                                                                sentence['score']))
                                            for sentence in sentences]

        standardized_response = AiDetectionDataClass(
            ai_score=score,
            items=items
        )

        return ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__plagia_detection(
        self, text: str, title: str = ""
    ) -> ResponseType[PlagiaDetectionDataClass]:

        payload = json.dumps({
            "api_key": self.api_key,
            "text": text
        })

        response = requests.request(
            "POST", f'{self.api_url}/plagiarism', headers=self.headers, data=payload)

        if response.status_code != 200:
            raise ProviderException(response.json(), code=response.status_code)

        original_response = response.json()
        results = original_response.get('results')

        if results is None:
            raise ProviderException(response.json())

        standardized_response = PlagiaDetectionDataClass(
            plagia_score=original_response["results_count"],
            items=[
                PlagiaDetectionItem(
                    text=''.join(result["excerpts"]),
                    candidates=[
                        PlagiaDetectionCandidate(
                            url=result["url"],
                            plagia_score=1,
                            prediction="plagiarized",
                            plagiarized_text=excerpt
                        ) for excerpt in result["excerpts"]
                    ]
                ) for result in results
            ]
        )

        return ResponseType[PlagiaDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
