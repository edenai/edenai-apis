import json
from typing import Dict, Tuple, Any, List, Optional, Literal

import requests

from edenai_apis.features import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text import (
    SummarizeDataClass,
    SyntaxAnalysisDataClass,
    AnonymizationDataClass,
    KeywordExtractionDataClass,
    InfosSyntaxAnalysisDataClass,
    InfosKeywordExtractionDataClass,
    SentimentAnalysisDataClass,
    SentimentEnum,
)
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationEntity,
)
from edenai_apis.features.text.anonymization.category import CategoryType
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException, LanguageException
from edenai_apis.utils.types import ResponseType
from .emvista_tags import tags


class EmvistaApi(ProviderInterface, TextInterface):
    provider_name = "emvista"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://pss-api.prevyo.com/pss/api/v1/"

    def _prepare_request(
        self, language: str, text: str
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        if not language:
            raise LanguageException("Language not provided")

        files = {"text": text, "parameters": [{"name": "lang", "value": language}]}
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return files, headers

    def _make_request(
        self, endpoint: str, headers: Dict[str, str], files: Dict[str, Any]
    ):
        response = requests.post(
            f"{self.base_url}{endpoint}", headers=headers, json=files
        )
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal server error", code=500) from exc

        if response.status_code == 201:
            raise ProviderException("Input text is too long", code=response.status_code)

        if response.status_code != 200:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        result = original_response.get("result") or {}

        return original_response, result

    @staticmethod
    def _normalize_sentiment(rate: float) -> Literal["Positive", "Negative", "Neutral"]:
        if rate == "NaN":
            return SentimentEnum.NEUTRAL.value
        if rate > 0:
            return SentimentEnum.POSITIVE.value
        if rate < 0:
            return SentimentEnum.NEGATIVE.value
        return SentimentEnum.NEUTRAL.value

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[SummarizeDataClass]:
        files, headers = self._prepare_request(language, text)

        original_response, result = self._make_request("summarizer", headers, files)

        standardized_response_list = result.get("sentences", [])

        level_items = [
            element for element in standardized_response_list if element["level"] == 10
        ]
        result = "".join([element["value"] for element in level_items])

        standardized_response = SummarizeDataClass(result=result)

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__syntax_analysis(
        self, language: str, text: str, **kwargs
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        files, headers = self._prepare_request(language, text)

        original_response, result = self._make_request("parser", headers, files)

        items: List[InfosSyntaxAnalysisDataClass] = []
        for sentence in result.get("sentences", []):
            for word in sentence["tokens"]:
                if word["pos"] in tags:
                    gender = word.get("gender")
                    plural = word.get("plural")

                    other = {
                        "gender": gender,
                        "plural": plural,
                        "mode": word.get("mode"),
                        "infinitive": None,
                    }
                    items.append(
                        InfosSyntaxAnalysisDataClass(
                            word=word["form"],
                            tag=tags[word["pos"]],
                            lemma=word["lemma"],
                            others=other,
                            importance=None,
                        )
                    )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__anonymization(
        self,
        text: str,
        language: str,
        model: Optional[str] = None,
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[AnonymizationDataClass]:
        files, headers = self._prepare_request(language, text)

        original_response, result = self._make_request("anonymizer", headers, files)

        entities: List[AnonymizationEntity] = []
        new_text = text

        for entity in result.get("namedEntities", []):
            classification = CategoryType.choose_category_subcategory(
                entity["tags"][0].split("/")[-1]
            )
            tmp_new_text = new_text[0 : entity["start"]] + "*" * (
                entity["end"] - entity["start"]
            )
            tmp_new_text += new_text[entity["end"] :]
            new_text = tmp_new_text
            entities.append(
                AnonymizationEntity(
                    content=entity["value"],
                    original_label=entity["tags"][0],
                    offset=entity["start"],
                    length=len(entity["value"]),
                    confidence_score=None,
                    category=classification["category"],
                    subcategory=classification["subcategory"],
                )
            )

        # Return standardized response
        standardized_response = AnonymizationDataClass(
            result=new_text,
            entities=entities,
        )

        result = ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        files, headers = self._prepare_request(language, text)

        original_response, result = self._make_request("opinions", headers, files)

        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=EmvistaApi._normalize_sentiment(
                result.get("globalScore", 0)
            ),
            general_sentiment_rate=(
                abs(result.get("globalScore", 0))
                if result.get("globalScore") != "NaN"
                else 0
            ),
        )

        result = ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        parameters:
          languages: string
          text: string

        return:
          ResponseType(
            original_response: {},
            standardized_response: Keyword_extraction(text: str)
          )
        """
        files, headers = self._prepare_request(language, text)

        original_response, result = self._make_request("keywords", headers, files)

        # Standardize response
        items: List[InfosKeywordExtractionDataClass] = []
        keywords = result.get("keywords") or []
        for keyword in keywords:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=keyword["value"], importance=float(keyword["score"]) * 0.25
                )
            )
        standardized_response = KeywordExtractionDataClass(items=items)

        # Return standardized response
        result = ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
