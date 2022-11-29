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
from modernmt import ModernMT


class ModernmtApi(
    ProviderApi,
    Translation,
):

    provider_name = "modernmt"
    
    def __init__(self):
        super().__init__()
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.client = ModernMT(self.api_settings["api_key"])
        
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        
        # Api output
        output = self.client.detect_language(text)
        response = output.__dict__

        # Create output TextDetectLanguage object
        # Analyze response
        items: Sequence[InfosLanguageDetectionDataClass] = []
        items.append(
                    InfosLanguageDetectionDataClass(
                        language=response['detectedLanguage'],
                    )
                )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass](
            original_response=response, standarized_response=standarized_response
        )
        
    def translation__automatic_translation(self, source_language: str, target_language: str, text: str
                                           )-> ResponseType[AutomaticTranslationDataClass]:
        
        #Api output
        output = self.client.translate(source_language, target_language, text)
        response = output.__dict__

        # if response.status_code != 200:
        #     raise ProviderException(message=original_response['message'], code=response.status_code)


        standarized_response = AutomaticTranslationDataClass(
            text=response['translation']
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standarized_response=standarized_response
        )
