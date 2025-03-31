from typing import Sequence, Optional

from ibm_watson.natural_language_understanding_v1 import (
    Features,
    SentimentOptions,
    SyntaxOptions,
    SyntaxOptionsTokens,
)

from edenai_apis.features.text import (
    InfosSyntaxAnalysisDataClass,
    SegmentSentimentAnalysisDataClass,
    SentimentAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.syntax_analysis.syntax_analysis_dataclass import (
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.types import ResponseType

from .config import tags
from .ibm_helpers import handle_ibm_call


class IbmTextApi(TextInterface):
    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
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

    def text__syntax_analysis(
        self, language: str, text: str, **kwargs
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(
                syntax=SyntaxOptions(
                    sentences=True,
                    tokens=SyntaxOptionsTokens(lemma=True, part_of_speech=True),
                )
            ),
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Getting syntax detected of word and its score of confidence
        for keyword in response["syntax"]["tokens"]:
            tag_ = tags[keyword["part_of_speech"]]
            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=keyword["text"],
                    importance=None,
                    others=None,
                    tag=tag_,
                    lemma=keyword.get("lemma"),
                )
            )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        return ResponseType[SyntaxAnalysisDataClass](
            original_response=response, standardized_response=standardized_response
        )
