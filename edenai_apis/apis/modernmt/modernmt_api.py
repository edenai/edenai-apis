from typing import Dict, Sequence, Optional

import requests

from edenai_apis.features import ProviderInterface, TranslationInterface
from edenai_apis.features.translation import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
    AutomaticTranslationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType


class ModernmtApi(ProviderInterface, TranslationInterface):
    provider_name = "modernmt"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.header = {"MMT-ApiKey": self.api_settings["api_key"]}
        self.url = "https://api.modernmt.com/translate"

    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = requests.get(
            url=f"{self.url}/detect", headers=self.header, data={"q": text}
        )

        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(
                message=original_response["error"]["message"], code=response.status_code
            )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        items.append(
            InfosLanguageDetectionDataClass(
                language=original_response["data"]["detectedLanguage"],
                display_name=get_language_name_from_code(
                    isocode=original_response["data"]["detectedLanguage"]
                ),
                confidence=None,
            )
        )

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items),
        )

    def translation__automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[AutomaticTranslationDataClass]:
        data = {
            "source": source_language,
            "target": target_language,
            "q": text,
        }

        # Api output
        output = requests.get(self.url, headers=self.header, data=data)
        response = output.json()

        # Handle error
        if response["status"] != 200:
            raise ProviderException(
                message=response["error"]["message"], code=response["status"]
            )

        standardized_response = AutomaticTranslationDataClass(
            text=response["data"]["translation"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standardized_response=standardized_response
        )
