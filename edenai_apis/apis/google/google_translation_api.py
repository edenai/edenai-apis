from typing import Sequence

from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection import (
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType

from google.protobuf.json_format import MessageToDict


class GoogleTranslationApi(TranslationInterface):
    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        # Getting response
        client = self.clients["translate"]
        parent = f"projects/{self.project_id}/locations/global"

        response = client.translate_text(
            parent=parent,
            contents=[text],
            mime_type="text/plain",  # mime types: text/plain, text/html
            source_language_code=source_language,
            target_language_code=target_language,
        )

        # Analyze response
        # Getting the translated text
        data = response.translations
        res = data[0].translated_text
        std: AutomaticTranslationDataClass
        if res != "":
            std = AutomaticTranslationDataClass(text=res)
        else:
            raise ProviderException("Empty Text was returned")
        return ResponseType[AutomaticTranslationDataClass](
            original_response=MessageToDict(response._pb), standardized_response=std
        )

    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = self.clients["translate"].detect_language(
            parent=f"projects/{self.project_id}/locations/global",
            content=text,
            mime_type="text/plain",
        )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        for language in response.languages:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=language.language_code,
                    display_name=get_language_name_from_code(
                        isocode=language.language_code
                    ),
                    confidence=language.confidence,
                )
            )
        return ResponseType[LanguageDetectionDataClass](
            original_response=MessageToDict(response._pb),
            standardized_response=LanguageDetectionDataClass(items=items),
        )
