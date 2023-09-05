import requests

from edenai_apis.apis.nlpcloud.utils import Iso_to_code
from edenai_apis.features import ProviderInterface, TextInterface
from typing import Dict, Sequence

from edenai_apis.features.text import KeywordExtractionDataClass, InfosKeywordExtractionDataClass, \
    SentimentAnalysisDataClass, SegmentSentimentAnalysisDataClass, CodeGenerationDataClass, \
    NamedEntityRecognitionDataClass, InfosNamedEntityRecognitionDataClass
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
        self.url_sentiment_analysis = (
            "https://api.nlpcloud.io/v1/distilbert-base-uncased-finetuned-sst-2-english/sentiment")
        self.url_code_generation = ("https://api.nlpcloud.io/v1/gpu/finetuned-llama-2-70b/code-generation")
        self.url_basic = ("https://api.nlpcloud.io/v1/")

    def text__spell_check(
            self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        if language == "en":
            url = self.url_spell_check
        else:
            url = self.url_basic + f"gpu/{Iso_to_code.get(language)}/finetuned-llama-2-70b/gs-correction"
        response = requests.post(url=url, json={"text": text},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        original_response = response.json()
        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        data = original_response["correction"]
        items: Sequence[SpellCheckItem] = [SpellCheckItem(
            text=text,
            offset=0,
            length=len(data),
            type=None,
            suggestions=[SuggestionItem(suggestion=data, score=None)]
        )]
        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )

    def text__keyword_extraction(
            self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        if language == "en":
            url = self.url_keyword_extraction
        else:
            url = self.url_basic + f"gpu/{Iso_to_code.get(language)}/finetuned-llama-2-70b/kw-kp-extraction"
        response = requests.post(url=url, json={"text": text},
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
        items: Sequence[SegmentSentimentAnalysisDataClass] = []
        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=original_response["scored_labels"][0]["label"],
            general_sentiment_rate=float(abs(original_response["scored_labels"][0]["score"])),
            items=items
        )
        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response, standardized_response=standardized_response
        )

    def text__code_generation(
            self, instruction: str, temperature: float, max_tokens: int, prompt: str = ""
    ) -> ResponseType[CodeGenerationDataClass]:

        response = requests.post(url=self.url_code_generation, json={"instruction": instruction},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        original_response = response.json()
        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        standardized_response = CodeGenerationDataClass(
            generated_text=original_response["generated_code"]
        )
        return ResponseType[CodeGenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__named_entity_recognition(
            self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        url_model = "news"
        if language == "en" or language == "zh":
            url_model = "web"
        url = self.url_basic + f"{language}_core_{url_model}_lg/entities"
        response = requests.post(url=url, json={"text": text},
                                 headers={"Content-Type": "application/json", "authorization": f"Token {self.api_key}"})
        try:
            original_response = response.json()
        except:
            raise ProviderException(response.text, code=response.status_code)

        if not response.ok:
            raise ProviderException(original_response, code=response.status_code)
        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for entity in original_response["entities"]:
            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=entity["text"],
                    category=entity["type"],
                    importance=None
                )
            )
        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=NamedEntityRecognitionDataClass(items=items)
        )
