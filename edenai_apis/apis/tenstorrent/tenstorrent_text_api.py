import requests
from typing import List, Optional

from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    KeywordExtractionDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.question_answer.question_answer_dataclass import (
    QuestionAnswerDataClass,
)
from edenai_apis.features.text.topic_extraction.topic_extraction_dataclass import (
    TopicExtractionDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TenstorrentTextApi(TextInterface):
    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        base_url = "https://keyword-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/keyword_extraction"
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

        standardized_response = KeywordExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        base_url = "https://sentiment-analysis--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/sentiment_analysis"
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

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
    ) -> ResponseType[QuestionAnswerDataClass]:
        base_url = "https://question-answer--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/question_answer"
        payload = {
            "text": texts[0],
            "question": question,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
            original_response.raise_for_status()
        except Exception as exc:
            raise ProviderException(original_response.text)

        original_response = original_response.json()

        # Check for errors
        self.check_for_errors(original_response)

        standardized_response = QuestionAnswerDataClass(
            answers=[original_response["answer"]]
        )
        return ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__topic_extraction(
        self, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        base_url = "https://topic-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/topic_extraction"
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

        standardized_response = TopicExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def check_for_errors(self, response):
        if "message" in response:
            raise ProviderException(response["message"])
