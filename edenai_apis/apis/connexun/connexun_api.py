from typing import Optional, Sequence
import requests

from edenai_apis.features import ProviderApi, Text
from edenai_apis.features.text import (
    SentimentAnalysisDataClass,
    SummarizeDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class ConnexunApi(ProviderApi, Text):
    provider_name = "connexun"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api"]
        self.base_url = self.api_settings["url"]

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        # Prepare request
        files = {"text": text}
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}text-analysis/sentiment"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        # Check errors from API
        if isinstance(original_response, dict) and original_response.get("message"):
            raise ProviderException(original_response["message"])

        # Return standarized response
        items: Sequence[Items] = []
        items.append(
            Items(
                sentiment=original_response.get("Sentiment"),
                sentiment_rate=original_response.get("Value"),
            )
        )
        standarized_response = SentimentAnalysisDataClass(items=items)

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: Optional[str]
    ) -> ResponseType[SummarizeDataClass]:
        # Prepare request
        files = {"text": text, "output_sentences": output_sentences}
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}text-analysis/summarize"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        # Check errors from API
        if isinstance(original_response, dict) and original_response.get("message"):
            raise ProviderException(original_response["message"])

        # Return standarized response
        standarized_response = SummarizeDataClass(result=original_response["summary"])
        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result
