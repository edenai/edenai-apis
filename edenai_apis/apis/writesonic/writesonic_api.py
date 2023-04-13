from typing import Optional
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    SummarizeDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
import json


class WritesonicApi(ProviderInterface, TextInterface):
    provider_name = "writesonic"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.api_key
        }


    def text__summarize(
        self, text: str,
        output_sentences: int,
        language: str, 
        model: str = None
        ) -> ResponseType[SummarizeDataClass]:
        url = f"https://api.writesonic.com/v2/business/content/summary?engine=premium&language={language}"
        payload = {
            "article_text": text,
            }
        
        try:
            original_response = requests.post(url, json=payload, headers= self.headers).json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error") from exc
        
        if "detail" in original_response:
            raise ProviderException(original_response['detail'])
        
        standardized_response = SummarizeDataClass(result=original_response[0].get("summary", {}))
        
        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )
        