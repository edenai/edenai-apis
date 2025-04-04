from typing import List, Optional

import requests

from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    KeywordExtractionDataClass,
)
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.question_answer.question_answer_dataclass import (
    QuestionAnswerDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.topic_extraction.topic_extraction_dataclass import (
    TopicExtractionDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

from edenai_apis.features.text.generation import GenerationDataClass

class TenstorrentTextApi(TextInterface):
    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        base_url = "https://keyword-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/keyword_extraction"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(
                message=original_response.text, code=original_response.status_code
            )

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = KeywordExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        base_url = "https://sentiment-analysis--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/sentiment_analysis"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(
                message=original_response.text, code=original_response.status_code
            )

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

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
        **kwargs,
    ) -> ResponseType[QuestionAnswerDataClass]:
        base_url = "https://question-answer--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/question_answer"
        payload = {
            "text": texts[0],
            "question": question,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(
                message=original_response.text, code=original_response.status_code
            )

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = QuestionAnswerDataClass(
            answers=[original_response["answer"]]
        )
        return ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__named_entity_recognition(
        self, text: str, language: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        base_url = "https://named-entity-recognition--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/named_entity_recognition"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(message=original_response.text, code=original_response.status_code)

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = NamedEntityRecognitionDataClass(
            items=original_response["items"]
        )
        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__topic_extraction(
        self, text: str, language: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[TopicExtractionDataClass]:
        base_url = "https://topic-extraction--eden-ai.workload.tenstorrent.com"
        url = f"{base_url}/predictions/topic_extraction"
        payload = {
            "text": text,
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=str(exc), code=500)
        if original_response.status_code != 200:
            raise ProviderException(
                message=original_response.text, code=original_response.status_code
            )

        status_code = original_response.status_code
        original_response = original_response.json()

        # Check for errors
        self.__check_for_errors(original_response, status_code)

        standardized_response = TopicExtractionDataClass(
            items=original_response["items"]
        )
        return ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
    
        
    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[GenerationDataClass]:
        payload = {
            "model": model,
            "prompt": text,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self.client.completions.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        # Standardize the response
        generated_text = response.choices[0].text

        standardized_response = GenerationDataClass(
            generated_text=generated_text,
        )

        return ResponseType[GenerationDataClass](
            original_response=response.to_dict(),
            standardized_response=standardized_response,
        )
    

    def __check_for_errors(self, response, status_code=None):
        if "message" in response:
            raise ProviderException(response["message"], code=status_code)
        