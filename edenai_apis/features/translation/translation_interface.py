from abc import ABC, abstractmethod

from edenai_apis.features.translation.automatic_translation.automatic_translation_dataclass import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.document_translation import (
    DocumentTranslationDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import (
    LanguageDetectionDataClass,
)
from edenai_apis.utils.types import ResponseType


class TranslationInterface:
    @abstractmethod
    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        """
        Translate a text

        Args:
            text (str): text to translate
            source_language (str): text's language code in ISO format
            target_language (str): to which language to translate text

        Note:
            for some providers, `source_language` can automatically detected
        """
        raise NotImplementedError

    @abstractmethod
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        """
        Detect language of a given text

        Args:
            text (str): text to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def translation__document_translation(
        self,
        file: str,
        file_type: str,
        source_language: str,
        target_language: str,
        file_url: str = "",
    ) -> ResponseType[DocumentTranslationDataClass]:
        """
        Translate a document

        Args:
            file (str): text to translate
            source_language (str): text's language code in ISO format
            target_language (str): to which language to translate text

        Note:
            for some providers, `source_language` can automatically detected
        """
        raise NotImplementedError
