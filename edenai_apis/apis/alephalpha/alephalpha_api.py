from typing import Dict

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AlephAlphaApi(ProviderInterface, TextInterface):
    provider_name = "alephalpha"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url_basic = "https://api.aleph-alpha.com"
        self.url_summarise = "https://api.aleph-alpha.com/summarize"
    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: str,
    ) -> ResponseType[SummarizeDataClass]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "document": {
                "text": text
            }
        }
        response = requests.post(url=self.url_summarise, headers=headers, json=payload)
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)
        original_response = response.json()
        standardized_response = SummarizeDataClass(
            result=original_response.get("summary", {})
        )
        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )


