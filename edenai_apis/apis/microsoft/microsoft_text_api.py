import sys
from collections import defaultdict
from time import sleep
from typing import Dict, Optional, Sequence

import requests
from edenai_apis.features.text import (
    InfosKeywordExtractionDataClass,
    InfosNamedEntityRecognitionDataClass,
    KeywordExtractionDataClass,
    NamedEntityRecognitionDataClass,
    SentimentAnalysisDataClass,
    SummarizeDataClass,
)
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SegmentSentimentAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class MicrosoftTextApi(TextInterface):
    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                TextNamedEntityRecognition Object that contains
        the entities and their importances
        """

        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "EntityRecognition",
                    "parameters": {"modelVersion": "latest"},
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc

        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for ent in data["results"]["documents"][0]["entities"]:
            entity = ent["text"]
            importance = ent["confidenceScore"]
            entity_type = ent["category"].upper()
            if entity_type == "DATETIME":
                entity_type = "DATE"

            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=entity,
                    importance=importance,
                    category=entity_type,
                )
            )

        standardized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=data, standardized_response=standardized_response
        )

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
    ) -> ResponseType[SummarizeDataClass]:

        """
        :param text:        String that contains input text
        :return:            String that contains output result
        """

        response = requests.post(
            self.url["summarization"],
            headers=self.headers["text"],
            json={
                "analysisInput": {"documents": [{"id": "1", "text": text}]},
                "tasks": {
                    "extractiveSummarizationTasks": [
                        {
                            "parameters": {
                                "model-version": "latest",
                                "sentenceCount": output_sentences,
                                "sortBy": "Offset",
                            }
                        }
                    ]
                },
            },
        )

        if response.status_code != 202:
            err = response.json().get("error", {})
            details = err.get("details", [defaultdict])[0]
            error_msg = details.get("message", "Microsoft Azure couldn't create job")
            raise ProviderException(error_msg)

        get_url = response.headers.get("operation-location")
        if get_url is None:
            raise ProviderException("Microsoft Azure couldn't create job")

        get_response = requests.get(url=get_url, headers=self.headers["text"])
        if get_response.status_code != 200:
            err = get_response.json().get("error", {})
            error_msg = err.get("message", "Microsoft Azure couldn't fetch job")
            raise ProviderException(error_msg)

        data = get_response.json()
        wait_time = 0
        summary = ""
        while wait_time < 60:  # Wait for the answer from provider
            if data["status"] == "succeeded":
                sentences = data["tasks"]["extractiveSummarizationTasks"][0]["results"][
                    "documents"
                ][0]["sentences"]
                summary = " ".join([sentence["text"] for sentence in sentences])
                break
            sleep(6)
            wait_time += 6
            get_response = requests.get(url=get_url, headers=self.headers["text"])
            data = get_response.json()

        standardized_response = SummarizeDataClass(result=summary)

        return ResponseType[SummarizeDataClass](
            original_response=data, standardized_response=standardized_response
        )

    def text__anonymization(
        self, text: str, language: str
    ) -> ResponseType[AnonymizationDataClass]:
        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "PiiEntityRecognition",
                    "parameters": {
                        "modelVersion": "latest",
                    },
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc

        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(original_response, response.status_code)

        standardized_response = AnonymizationDataClass(
            result=original_response["results"]["documents"][0]["redactedText"]
        )
        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        """
        :param language:    String that contains language code
        :param text:        String that contains the text to analyse
        :return:            TextSentimentAnalysis Object that contains sentiments and their rates
        """
        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "SentimentAnalysis",
                    "parameters": {
                        "modelVersion": "latest",
                    },
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc

        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[SegmentSentimentAnalysisDataClass] = []

        # Getting the explicit label and its score of image
        default_dict = defaultdict(lambda: None)
        sentences = (
            data.get("results", default_dict)
            .get("documents", [default_dict])[0]
            .get("sentences")
        )
        if sentences:
            for sentence in sentences:
                best_sentiment = {
                    "sentiment": None,
                    "rate": 0,
                }
                for sentiment, value in sentence["confidenceScores"].items():
                    if best_sentiment["rate"] < value:
                        best_sentiment["sentiment"] = sentiment
                        best_sentiment["rate"] = value

                items.append(
                    SegmentSentimentAnalysisDataClass(
                        segment=sentence["text"],
                        sentiment=best_sentiment["sentiment"],
                        sentiment_rate=best_sentiment["rate"],
                    )
                )

        best_general_sentiment = {"sentiment": None, "rate": 0}
        for sentiment, value in data["results"]["documents"][0][
            "confidenceScores"
        ].items():
            if best_general_sentiment["rate"] < value:
                best_general_sentiment["sentiment"] = sentiment
                best_general_sentiment["rate"] = value

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_general_sentiment["sentiment"],
            general_sentiment_rate=best_general_sentiment["rate"],
            items=items,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=data, standardized_response=standarize
        )

    def _check_microsoft_error(self, data: Dict):
        if data.get("error", {}).get("message"):
            raise Exception(data["error"]["message"])

    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        :param language:    String that contains language code
        :param text:        String that contains the text to analyse
        :return:            TextKeywordExtraction Object that contains the Key phrases
        """

        try:
            response = requests.post(
                f"{self.url['text']}",
                headers=self.headers["text"],
                json={
                    "kind": "KeyPhraseExtraction",
                    "parameters": {"modelVersion": "latest"},
                    "analysisInput": {
                        "documents": [{"id": "1", "language": language, "text": text}]
                    },
                },
            )
        except Exception as exc:
            raise ProviderException(f"Unexpected error! {sys.exc_info()[0]}") from exc
        data = response.json()
        self._check_microsoft_error(data)

        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in data["results"]["documents"][0]["keyPhrases"]:
            items.append(InfosKeywordExtractionDataClass(keyword=key_phrase))

        standardized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=data, standardized_response=standardized_response
        )
