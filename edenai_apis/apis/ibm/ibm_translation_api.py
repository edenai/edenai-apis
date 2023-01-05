from typing import Sequence
import re

from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException


class IbmTranslationApi(TranslationInterface):
    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:

        try:
            response = (
                self.clients["translation"]
                .translate(text=text, source=source_language, target=target_language)
                .get_result()
            )
        except Exception as excp:
            error_message = str(excp)
            try:
                language_detection_error = re.search("Error:(.)*language,", error_message).group()[:-1]
                raise ProviderException(language_detection_error)
            except AttributeError:
                raise ProviderException(error_message)

        # Create output TextAutomaticTranslation object
        standardized: AutomaticTranslationDataClass

        # Getting the translated text
        for translated_text in response["translations"]:
            standardized = AutomaticTranslationDataClass(
                text=translated_text["translation"]
            )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response["translations"],
            standardized_response=standardized,
        )

    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = self.clients["translation"].identify(text).get_result()
        items: Sequence[InfosLanguageDetectionDataClass] = []

        for lang in response["languages"]:
            if lang["confidence"] > 0.2:
                items.append(
                    InfosLanguageDetectionDataClass(
                        language=lang["language"],
                        display_name=get_language_name_from_code(
                            isocode=lang["language"]
                        ),
                        confidence=lang["confidence"],
                    )
                )

        return ResponseType[LanguageDetectionDataClass](
            original_response=response,
            standardized_response=LanguageDetectionDataClass(items=items),
        )
