from http import HTTPStatus
from typing import Sequence, Optional

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

    def _raise_on_error(self, response: requests.Response) -> None:
        """
        Raise ProviderException if the response is not successful
        """
        status = response.status_code
        if status >= 500:
            # we pass generic http error instead of returning whole html response
            message = HTTPStatus(status).phrase
            raise ProviderException(message=message, code=status)
        if status >= 400:
            data = response.json()
            error = data.get("error", {}) or {}
            error_message = error.get("message", "")
            raise ProviderException(error_message, code=status)

    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
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

        self._raise_on_error(response)
        data = response.json()

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
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: Optional[str] = None,
        **kwargs,
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
        self._raise_on_error(response)
        data = response.json()

        # Create output TextAutomaticTranslation object
        standardized_response = AutomaticTranslationDataClass(
            text=data[0]["translations"][0]["text"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=data, standardized_response=standardized_response
        )
