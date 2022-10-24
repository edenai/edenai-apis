from typing import Optional
import requests
from edenai_apis.features import ProviderApi, Text
from edenai_apis.features.text import SummarizeDataClass
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.types import ResponseType



class MeaningcloudApi(ProviderApi, Text):
    provider_name = "meaningcloud"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.url = self.api_settings["url"]

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: Optional[str]
    ) -> ResponseType[SummarizeDataClass]:
        data = {
            "key": self.api_key,
            "txt": text,
            "sentences": output_sentences,
        }
        response = requests.post(self.url, data=data)

        original_response = response.json()

        standarized_response = SummarizeDataClass(result=original_response["summary"])

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )
        return result
