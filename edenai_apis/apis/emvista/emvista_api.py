from typing import Optional, Sequence

import requests
from edenai_apis.features import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text import (
    AnonymizationDataClass,
    InfosKeywordExtractionDataClass,
    InfosSyntaxAnalysisDataClass,
    KeywordExtractionDataClass,
    SentimentAnalysisDataClass,
    SummarizeDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentEnum,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.types import ResponseType

from .emvista_tags import tags


class EmvistaApi(ProviderInterface, TextInterface):
    provider_name = "emvista"
    base_url = "https://pss-api.prevyo.com/pss/api/v1/"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: Optional[str] = None
    ) -> ResponseType[SummarizeDataClass]:

        #check language
        if not language:
            raise LanguageException("Language not provided")

        # Prepare request
        files = {"text": text, "parameters": [{"name": "lang", "value": language}]}
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}summarizer"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        # Check errors from API
        if response.status_code != 200:
            raise ProviderException(original_response["message"])

        # Return standardized response
        standardized_response_list = original_response["result"].get("sentences", [])
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
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:

        #check language
        if not language:
            raise LanguageException("Language not provided")
        # Prepare request
        files = {"text": text, "parameters": [{"name": "lang", "value": language}]}
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}parser"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        items: Sequence[InfosSyntaxAnalysisDataClass] = []
        for sentence in original_response["result"].get("sentences", []):
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
                        )
                    )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__anonymization(
        self, text: str, language: str
    ) -> ResponseType[AnonymizationDataClass]:
        # Prepare request
        files = {"text": text, "parameters": [{"name": "lang", "value": language}]}
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}anonymizer"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        # Check errors from API
        if response.status_code != 200:
            raise ProviderException(original_response["message"])

        # Return standardized response
        standardized_response = AnonymizationDataClass(
            result=original_response["result"]["annotatedValue"]
        )

        result = ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def _normalize_sentiment(self, rate: float) -> str:
        if rate == 'NaN':
            return SentimentEnum.NEUTRAL.value
        if rate > 0:
            return SentimentEnum.POSITIVE.value
        if rate < 0:
            return SentimentEnum.NEGATIVE.value
        return SentimentEnum.NEUTRAL.value

    def text__sentiment_analysis(
        self,
        language: str,
        text: str
        ) -> ResponseType[SentimentAnalysisDataClass]:
        # check language
        if not language:
            raise LanguageException("Language not provided")
        # Prepare request
        files = {
            "text": text,
            "parameters": [{
                "name": "lang",
                "value": language
            }]
        }
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        url = f"{self.base_url}opinions"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        print(original_response['result']['globalScore'])

        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=self._normalize_sentiment(original_response['result']['globalScore']),
            general_sentiment_rate=abs(original_response['result']['globalScore']) if original_response['result']['globalScore'] != 'NaN' else 0
        )
        
        result = ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
        return result

    def text__keyword_extraction(
        self, language: str, text: str
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
        #check language
        if not language:
            raise LanguageException("Language not provided")
        # Prepare request
        files = {"text": text, "parameters": [{"name": "lang", "value": language}]}
        headers = {
            "Poa-Token": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}keywords"

        # Send request to API
        response = requests.post(url, headers=headers, json=files)
        original_response = response.json()

        # Check error from API
        if response.status_code != 200:
            raise ProviderException(original_response["message"])

        # Standardize response
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for keyword in original_response["result"]["keywords"]:
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
