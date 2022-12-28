from typing import Sequence
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SegmentSentimentAnalysisDataClass,
    SentimentEnum
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType

from .lettria_tags import tags


class LettriaApi(ProviderInterface, TextInterface):
    provider_name = "lettria"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.lettria.com/"
        self.headers = {
            "Authorization": self.api_key,
        }

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        original_response = requests.post(
            url=self.url, headers=self.headers, json={"text": text}
        ).json()

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for value in original_response["sentences"]:
            item = value.get("ml_ner", [])
            for entity in item:
                items.append(
                    InfosNamedEntityRecognitionDataClass(
                        entity=entity["source"],
                        importance=None,
                        category=entity["type"],
                    )
                )

        standardized_response = NamedEntityRecognitionDataClass(items=items)

        result = ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result


    def _normalize_sentiment(self, rate: float) -> SentimentEnum:
        if rate > 0:
            return SentimentEnum.POSITIVE
        if rate < 0:
            return SentimentEnum.NEGATIVE
        return SentimentEnum.NEUTRAL


    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        original_response = requests.post(
            url=self.url, headers=self.headers, json={"text": text}
        ).json()

        items = []
        for sentence in original_response['sentences']:
            score = sentence['sentiment']['subsentences'][0]['values']['total']
            sentiment = self._normalize_sentiment(score).value
            items.append(SegmentSentimentAnalysisDataClass(
                segment=sentence['sentiment']['subsentences'][0]['sentence'],
                sentiment=sentiment,
                sentiment_rate=abs(sentence['sentiment']['subsentences'][0]['values']['total'])
            ))

        sentiment: str = self._normalize_sentiment(original_response['sentiment']).value
        sentiment_rate: float = abs(original_response["sentiment"])

        standarize = SentimentAnalysisDataClass(
            general_sentiment=sentiment,
            general_sentiment_rate=sentiment_rate,
            items=items
        )

        result = ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standarize,
        )
        return result

    def text__syntax_analysis(
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:

        original_response = requests.post(
            url=self.url, headers=self.headers, json={"text": text}
        ).json()

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        for sentence in original_response["sentences"]:
            for word in sentence["detail"]:
                if word["tag"] in tags:
                    gender = None
                    plural = None
                    # lemmatizer can be a dict or a list.
                    # gender and plural are only available if the lemmatizer is a dict
                    if isinstance(word.get("lemmatizer"), dict):
                        if word.get("lemmatizer", {}).get("gender", {}).get("female"):
                            gender = "feminine"
                        elif (
                            not word.get("lemmatizer", {})
                            .get("gender", {})
                            .get("female")
                        ):
                            gender = "masculine"
                        if word.get("lemmatizer", {}).get("gender", {}).get("plural"):
                            plural = "plural"
                        elif (
                            not word.get("lemmatizer", {})
                            .get("gender", {})
                            .get("plural")
                        ):
                            plural = "singular"
                    other = {
                        "gender": gender,
                        "plural": plural,
                        "mode": None,
                        "infinitive": word.get("infinit"),
                    }
                    items.append(
                        InfosSyntaxAnalysisDataClass(
                            word=word["source"],
                            tag=tags[word["tag"]],
                            lemma=word["lemma"],
                            others=other,
                        )
                    )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result
