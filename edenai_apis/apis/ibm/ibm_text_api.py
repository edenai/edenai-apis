from typing import Sequence

from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

from edenai_apis.features.text import (
    SegmentSentimentAnalysisDataClass,
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.types import ResponseType

from .ibm_helpers import handle_ibm_call


class IbmTextApi(TextInterface):
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(sentiment=SentimentOptions()),
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)

        # Create output object
        items: Sequence[SegmentSentimentAnalysisDataClass] = []
        standarize = SentimentAnalysisDataClass(
            general_sentiment=response["sentiment"]["document"]["label"],
            general_sentiment_rate=float(
                abs(response["sentiment"]["document"]["score"])
            ),
            items=items,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standardized_response=standarize
        )
