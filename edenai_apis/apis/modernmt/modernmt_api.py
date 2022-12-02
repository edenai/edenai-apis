from edenai_apis.features.translation import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
    AutomaticTranslationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderApi, Translation
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from typing import Sequence
import json
import requests
from modernmt import ModernMT


class ModernmtApi(
    ProviderApi,
    Translation,
):

    provider_name = "modernmt"
    
    def __init__(self):
        super().__init__()
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.header = {
            'MMT-ApiKey' : self.api_settings["api_key"]
        }
        self.url = self.api_settings["url"]

        
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        
        data = {
            "q" : text,
        }
        
        # Api output
        output = requests.get("https://api.modernmt.com/translate/detect", 
                              headers=self.header, data= data)
        response=output.json()

        # Handle errors :
        if response['status'] != 200:
            raise ProviderException(message=response['error']['message'], code=response['status'])

        # Create output TextDetectLanguage object
        # Analyze response
        items: Sequence[InfosLanguageDetectionDataClass] = []
        items.append(
                    InfosLanguageDetectionDataClass(
                        language=response['data']['detectedLanguage'],
                    )
                )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass](
            original_response=response, standarized_response=standarized_response
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


        standarized_response = AutomaticTranslationDataClass(
            text=response['data']['translation']
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standarized_response=standarized_response
        )
