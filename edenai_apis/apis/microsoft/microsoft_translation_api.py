from typing import Sequence

import requests

from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.utils.conversion import add_query_param_in_url
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType


class MicrosoftTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text
    ) -> ResponseType[LanguageDetectionDataClass]:
        response = requests.post(
            url=f"{self.url['text']}",
            headers=self.headers["text"],
            json={
                "kind": "LanguageDetection",
                "parameters": {"modelVersion": "latest"},
                "analysisInput": {"documents": [{"id": "1", "text": text}]},
            },
        )

        data = response.json()
        if response.status_code != 200:
            raise ProviderException(
                message=data["error"]["message"], code=response.status_code
            )

        items: Sequence[InfosLanguageDetectionDataClass] = []
        for lang in data["results"]["documents"]:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=lang["detectedLanguage"]["iso6391Name"],
                    display_name=get_language_name_from_code(
                        isocode=lang["detectedLanguage"]["iso6391Name"]
                    ),
                    confidence=lang["detectedLanguage"]["confidenceScore"],
                )
            )
        return ResponseType[LanguageDetectionDataClass](
            original_response=data,
            standardized_response=LanguageDetectionDataClass(items=items),
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        """
        :param source_language:    String that contains language name of origin text
        :param target_language:    String that contains language name of origin text
        :param text:        String that contains input text to translate
        :return:            String that contains output result
        """

        # Create configuration dictionnary

        url = add_query_param_in_url(
            url=self.url["translator"],
            query_params={"from": source_language, "to": target_language},
        )

        body = [
            {
                "text": text,
            }
        ]
        # Getting response of API
        response = requests.post(url, headers=self.headers["translator"], json=body)
        data = response.json()

        if response.status_code >= 400:
            error = data.get("error", {}) or {}
            error_message = error.get("message", "")
            raise ProviderException(error_message, code = response.status_code)

        # Create output TextAutomaticTranslation object
        standardized_response = AutomaticTranslationDataClass(
            text=data[0]["translations"][0]["text"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=data, standardized_response=standardized_response
        )
