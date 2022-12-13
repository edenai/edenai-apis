from typing import Sequence
import requests

from edenai_apis.features.translation import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import LanguageKey, get_info_languages
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderApi, Translation
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class ModernmtApi(ProviderApi,Translation):
    provider_name = "modernmt"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.header = {
            'MMT-ApiKey' : self.api_settings["api_key"]
        }
        self.url = self.api_settings["url"]

    def translation__language_detection(self, text) -> ResponseType[LanguageDetectionDataClass]:
        response = requests.get(
            url="https://api.modernmt.com/translate/detect",
            headers=self.header,
            data={ "q" : text }
        )

        original_response=response.json()
        if response['status'] != 200:
            raise ProviderException(
                message=original_response['error']['message'],
                code=response['status']
            )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        items.append(
                    InfosLanguageDetectionDataClass(
                        language=get_info_languages(
                            key=LanguageKey.CODE,
                            value=response['data']['detectedLanguage']
                    ))
                )

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items)
        )
        
    def translation__automatic_translation(self, source_language: str, target_language: str, text: str
                                           )-> ResponseType[AutomaticTranslationDataClass]:
        data = {
            "source" : source_language,
            "target" : target_language,
            "q" : text,
        }
        
        # Api output
        output = requests.get('https://api.modernmt.com/translate',
                              headers = self.header, data=data)
        response = output.json()

        # Handle error 
        if response['status'] != 200:
            raise ProviderException(message=response['error']['message'], code=response['status'])


        standardized_response = AutomaticTranslationDataClass(
            text=response['data']['translation']
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standardized_response=standardized_response
        )
