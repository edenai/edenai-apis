from typing import Dict, Optional

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class MeaningcloudApi(ProviderInterface, TextInterface):
    provider_name = "meaningcloud"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.meaningcloud.com/summarization-1.0"

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[SummarizeDataClass]:
        data = {
            "key": self.api_key,
            "txt": text,
            "sentences": output_sentences,
        }
        response = requests.post(self.url, data=data)

        original_response = response.json()

        standardized_response = SummarizeDataClass(result=original_response["summary"])

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result
