from typing import Optional
import requests
from edenai_apis.features import TranslationInterface
from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection import (
    LanguageDetectionDataClass,
)
from edenai_apis.utils.types import ResponseType


class OpenaiTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = self.llm_client.language_detection(text=text, model=model)
        return response

    def translation__automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[AutomaticTranslationDataClass]:
        response = self.llm_client.automatic_translation(
            source_language=source_language,
            target_language=target_language,
            text=text,
            model=model,
            **kwargs,
        )
        return response
