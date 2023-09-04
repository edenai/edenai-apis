import requests

from edenai_apis.features import ProviderInterface, TextInterface
from typing import Dict, Sequence

from edenai_apis.features.text import KeywordExtractionDataClass, InfosKeywordExtractionDataClass, \
    SentimentAnalysisDataClass
from edenai_apis.features.text.spell_check import SpellCheckDataClass, SpellCheckItem, SuggestionItem
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import construct_word_list
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class NlpCloudApi(ProviderInterface, TextInterface):
    provider_name = "nlpcloud"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["subscription_key"]
        self.url_spell_check = ("https://api.nlpcloud.io/v1/gpu/finetuned-llama-2-70b/gs-correction")
        self.url_keyword_extraction = ("https://api.nlpcloud.io/v1/gpu/finetuned-llama-2-70b/kw-kp-extraction")
        self.url_sentiment_analysis = ("https://api.nlpcloud.io/v1/distilbert-base-uncased-finetuned-sst-2-english/sentiment" )

    # ATTENTION: items corriger
    def text__spell_check(
            self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        response = requests.post(url=self.url_spell_check, json={"text": text},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        original_response = response.json()
        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        data = original_response.get("correction", None)
        corrections = construct_word_list(text, data)
        items: Sequence[SpellCheckItem] = []
        for item in corrections:
            items.append(SpellCheckItem(
                text=text,
                offset=0,
                length=len(data),
                type=None,
                suggestions=[SuggestionItem(suggestion=items, score=1.0)]
            ))
        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )

    def text__keyword_extraction(
            self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        response = requests.post(url=self.url_keyword_extraction, json={"text": text},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        original_response = response.json()
        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for keyword in original_response["keywords_and_keyphrases"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=keyword, importance=None
                )
            )
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=KeywordExtractionDataClass(items=items)
        )
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        response = requests.post(url=self.url_sentiment_analysis, json={"text": text},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        original_response = response.json()
        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        standardized_response = SentimentAnalysisDataClass(

        )
