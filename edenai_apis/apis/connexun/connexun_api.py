import json
from typing import Dict, Optional

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    SentimentAnalysisDataClass,
    SummarizeDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class ConnexunApi(ProviderInterface, TextInterface):
    provider_name = "connexun"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api"]
        self.base_url = "https://api.connexun.com/"

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        files = {"text": text}
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}text-analysis/sentiment"

        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        if isinstance(original_response, dict) and original_response.get("message"):
            raise ProviderException(
                original_response["message"],
                code = response.status_code
            )

        general_sentiment = original_response.get("Sentiment")
        general_sentiment_rate = original_response.get("Value")

        if not general_sentiment and not general_sentiment_rate:
            raise ProviderException(
                "Provider has not found a sentiment of the text.",
                code = 200
            )

        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=original_response.get("Sentiment"),
            general_sentiment_rate=original_response.get("Value"),
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
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
        status_code = response.status_code
        try:
            original_response = response.json()
        except json.JSONDecodeError:
            raise ProviderException("Internal server error", code = status_code)
        if status_code != 200:
            raise ProviderException(response.json()["detail"], code = status_code)
        if original_response.get("Error"):
            raise ProviderException(original_response["Error"], code = status_code)

        # Check errors from API
        if isinstance(original_response, dict) and original_response.get("message"):
            raise ProviderException(original_response["message"], code = status_code)

        # Return standardized response
        standardized_response = SummarizeDataClass(
            result=original_response.get("summary", {})
        )
        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result
