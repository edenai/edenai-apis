from typing import Sequence, Optional

from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.features.translation.automatic_translation.automatic_translation_dataclass import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType


class AmazonTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = self.clients["text"].detect_dominant_language(Text=text)
        items: Sequence[InfosLanguageDetectionDataClass] = []
        for lang in response["Languages"]:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=lang["LanguageCode"],
                    display_name=get_language_name_from_code(
                        isocode=lang["LanguageCode"]
                    ),
                    confidence=lang["Score"],
                )
            )

        return ResponseType[LanguageDetectionDataClass](
            original_response=response,
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
        payload = {
            "Text": text,
            "SourceLanguageCode": source_language,
            "TargetLanguageCode": target_language,
        }
        response = handle_amazon_call(
            self.clients["translate"].translate_text, **payload
        )

        standardized: AutomaticTranslationDataClass = AutomaticTranslationDataClass(
            text=response["TranslatedText"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standardized_response=standardized
        )
