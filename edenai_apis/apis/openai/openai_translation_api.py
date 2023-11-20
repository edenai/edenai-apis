from typing import Sequence

import numpy as np
import requests

from edenai_apis.features import TranslationInterface
from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection import (
    LanguageDetectionDataClass,
    InfosLanguageDetectionDataClass,
)
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType
from .helpers import (
    get_openapi_response,
    construct_language_detection_context,
    construct_translation_context,
)


class OpenaiTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_language_detection_context(text)
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "temperature": 0,
            "logprobs": 1,
        }
        response = requests.post(
            url, json=payload, headers=self.headers
        )
        original_response = get_openapi_response(response)

        items: Sequence[InfosLanguageDetectionDataClass] = []

        score = np.exp(original_response["choices"][0]["logprobs"]["token_logprobs"][0])
        # replace are necessary to keep only language code
        isocode = original_response["choices"][0]["text"].replace(" ", "")
        items.append(
            InfosLanguageDetectionDataClass(
                language=isocode,
                display_name=get_language_name_from_code(isocode=isocode),
                confidence=float(score),
            )
        )

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items),
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_translation_context(text, source_language, target_language)
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        response = requests.post(
            url, json=payload, headers=self.headers
        )
        original_response = get_openapi_response(response)

        standardized = AutomaticTranslationDataClass(
            text=original_response["choices"][0]["text"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response, standardized_response=standardized
        )
