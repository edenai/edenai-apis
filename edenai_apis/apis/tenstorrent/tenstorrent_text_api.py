import requests

from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    KeywordExtractionDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TenstorrentTextApi(TextInterface):

	def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        url = f"{self.base_url}/predictions/keyword_extraction"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
            original_response.raise_for_status()
        except Exception as exc:
            raise ProviderException(original_response.text)

        original_response = original_response.json()

        # Check for errors
        self.check_for_errors(original_response)

        standardized_response = KeywordExtractionDataClass(items=original_response["items"])
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = f"{self.base_url}/predictions/sentiment_analysis"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
            original_response.raise_for_status()
        except Exception as exc:
            raise ProviderException(original_response.text)

        original_response = original_response.json()

        # Check for errors
        self.check_for_errors(original_response)

        # Create output response
        confidence = float(original_response["confidence"])
        prediction = original_response["prediction"]
        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=prediction,
            general_sentiment_rate=confidence,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def check_for_errors(self, response):
        if "message" in response:
            raise ProviderException(response["message"])
