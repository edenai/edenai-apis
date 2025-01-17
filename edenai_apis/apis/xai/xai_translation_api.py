
import requests
import json
from edenai_apis.features import TranslationInterface
from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.language_detection import (
    LanguageDetectionDataClass,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from .helpers import (
    get_openapi_response,
    construct_language_detection_context,
    construct_translation_context,
)


class XAiTranslationApi(TranslationInterface):
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_language_detection_context(text)
        json_output = {"items" : [{"language" : "isocode", "display_name": "language display name", "confidence" : 0.8}]}
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as language detection model capable of automatically identifying the language of a given text.
                you return a JSON object in this format : {json_output}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "grok-beta",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        languages = original_response["choices"][0]["message"]["content"]
        try:
            data = json.loads(languages)
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=data.get('items')),
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_translation_context(text, source_language, target_language)
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as an automatic translation model capable of translating text from one language to another with high accuracy and fluency.""",
            },
        )
        # Build the request
        payload = {
            "model": "grok-beta",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        translation = original_response["choices"][0]["message"]["content"]

        standardized = AutomaticTranslationDataClass(
            text=translation
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response, standardized_response=standardized
        )
