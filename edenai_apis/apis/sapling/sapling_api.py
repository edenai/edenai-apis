from typing import Dict
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
    SegmentSentimentAnalysisDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.utils.types import ResponseType

import requests
from pprint import pprint

from typing import cast

class SaplingApi(ProviderInterface, TextInterface):
    provider_name = "sapling"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api"]
        self.base_url = "https://api.sapling.com/"

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = "https://api.sapling.ai/api/v1/sentiment"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "key": self.api_key,
            "text": text
        }

        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()

        best_sentiment = {
            "general_sentiment": None,
            "general_sentiment_rate": 0,
            "items": [],
        }

        for i in range(len(response_json["sents"])):
            sentence = response_json["sents"][i]
            sentiment = response_json["results"][i][0][1]
            sentiment_rate = response_json["results"][i][0][0]

            if sentiment != "Mixed":
                if best_sentiment["general_sentiment_rate"] <= sentiment_rate:
                    best_sentiment["general_sentiment"] = sentiment
                    best_sentiment["general_sentiment_rate"] = sentiment_rate

                segment_sentiment = SegmentSentimentAnalysisDataClass(
                    segment=sentence,
                    sentiment=sentiment,
                    sentiment_rate=sentiment_rate
                )
                best_sentiment["items"].append(segment_sentiment)

        sentiment_analysis = SentimentAnalysisDataClass(
            general_sentiment=best_sentiment["general_sentiment"],
            general_sentiment_rate=best_sentiment["general_sentiment_rate"],
            items=best_sentiment["items"]
        )

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_sentiment["general_sentiment"],
            general_sentiment_rate=best_sentiment["general_sentiment_rate"],
            items=[]
        )

        return cast(
            ResponseType[SentimentAnalysisDataClass],
            ResponseType(
                original_response=response,
                standardized_response=standarize
            )
        )
