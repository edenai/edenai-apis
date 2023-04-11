from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import \
    SentimentAnalysisDataClass
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.types import ResponseType


class TenstorrentTextApi(TextInterface):
    def text__sentiment_analysis(
            self, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        # Getting response
        original_response = self.clients["text"].detect_sentiment(Text=text)

        # Standardize response to EdenAI specifications
        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=original_response["prediction"],
            general_sentiment_rate=original_response["confidence"],
            items=[],
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response, standardized_response=standardized_response
        )
