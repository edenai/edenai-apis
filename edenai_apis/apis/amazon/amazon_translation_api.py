from typing import Sequence

from botocore.exceptions import ParamValidationError
from edenai_apis.features.translation.automatic_translation.automatic_translation_dataclass import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.exception import LanguageException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType


class AmazonTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text
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
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        try:
            response = self.clients["translate"].translate_text(
                Text=text,
                SourceLanguageCode=source_language,
                TargetLanguageCode=target_language,
            )
        except ParamValidationError as exc:
            if "SourceLanguageCode" in str(exc):
                raise LanguageException(str(exc))

        standardized: AutomaticTranslationDataClass
        if response["TranslatedText"] != "":
            standardized = AutomaticTranslationDataClass(
                text=response["TranslatedText"]
            )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response, standardized_response=standardized.dict()
        )
