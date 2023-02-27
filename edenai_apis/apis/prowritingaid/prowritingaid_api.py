from http import HTTPStatus
from typing import Sequence
import requests
from edenai_apis.features.text.spell_check.spell_check_dataclass import SpellCheckDataClass, SpellCheckItem, SuggestionItem
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class ProWritingAidApi(ProviderInterface, TextInterface):
    provider_name = "prowritingaid"

    def __init__(self):
        api_settings = load_provider(ProviderDataEnum.KEY, provider_name=self.provider_name)
        self.api_key = api_settings['api_key']
        self.api_url = api_settings['url']

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "licenseCode": self.api_key
        }

    def text__spell_check(self, text: str, language: str) -> ResponseType[SpellCheckDataClass]:
        payload = {
            "text": text,
            "language": language,
            "style": "General",
            "reports": ["grammar"],
            "includeParagraphStat": False,
            "documentType": 0,
        }

        response = requests.post(url=f"{self.api_url}/text", headers=self.headers, json=payload)

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException(message="Internal Server Error", code=response.status_code)

        original_response = response.json()

        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise ProviderException(
                message=original_response.get("message", "Provider returned an unknown error"),
                code=response.status_code
            )

        items: Sequence[SpellCheckItem] = []
        for tag in original_response['Result']['Tags']:
            suggestions: Sequence[SuggestionItem] = []
            for suggestion in tag['suggestions']:
                suggestions.append(SuggestionItem(suggestion=suggestion))
            items.append(SpellCheckItem(
                text=tag['subcategory'],
                offset=tag['startPos'],
                length=tag['endPos'] - tag['startPos'],
                type=tag['hint'],
                suggestions=suggestions,
            ))

        standardized_response = SpellCheckDataClass(text=text, items=items)

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )

