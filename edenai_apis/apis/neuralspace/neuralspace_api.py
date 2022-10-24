from typing import Sequence
import requests

from edenai_apis.features import ProviderApi, Text, Translation
from edenai_apis.features.text import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    LanguageDetectionDataClass,
    InfosLanguageDetectionDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class NeuralSpaceApi(ProviderApi, Text, Translation):
    provider_name = "neuralspace"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api"]
        self.url = self.api_settings["url"]
        self.header = {
            "authorization": f"{self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        url = f"{self.url}ner/v1/entity"

        files = {"text": text, "language": language}

        response = requests.request("POST", url, json=files, headers=self.header)
        response = response.json()

        data = response["data"]

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        if len(data["entities"]) > 0:
            for entity in data["entities"]:

                items.append(
                    InfosNamedEntityRecognitionDataClass(
                        entity=entity["text"],
                        importance=None,
                        category=entity["type"],
                        url="",
                    )
                )

        standarized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=data, standarized_response=standarized_response
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}translation/v1/translate"

        files = {
            "text": text,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
        }

        response = requests.request("POST", url, json=files, headers=self.header)
        response = response.json()
        data = response["data"]

        standarized_response = AutomaticTranslationDataClass(
            text=data["translatedText"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=data, standarized_response=standarized_response
        )

    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        url = f"{self.url}language-detection/v1/detect"
        files = {"text": text}

        response = requests.request("POST", url, json=files, headers=self.header)
        response = response.json()

        items: Sequence[InfosLanguageDetectionDataClass] = []
        if len(response["data"]["detected_languages"]) > 0:
            for lang in response["data"]["detected_languages"]:
                confidence = float(lang["confidence"])
                if confidence > 0.1:
                    items.append(
                        InfosLanguageDetectionDataClass(
                            language=lang["language"], confidence=confidence
                        )
                    )

        standarized_response = LanguageDetectionDataClass(items=items)

        data = response["data"]

        return ResponseType[LanguageDetectionDataClass](
            original_response=data, standarized_response=standarized_response
        )
