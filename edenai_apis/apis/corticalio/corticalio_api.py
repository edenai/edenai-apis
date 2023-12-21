from typing import Dict

from edenai_apis.apis.corticalio.client import CorticalClient
from edenai_apis.apis.corticalio.helpers import normalize_keywords
from edenai_apis.features import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text import KeywordExtractionDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class CorticalioApi(
    ProviderInterface,
    TextInterface
):
    provider_name = "corticalio"

    def __init__(self, api_keys: Dict = None):
        api_keys = api_keys or {}
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.client = CorticalClient(self.api_settings)

    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        response = self.client.extract_keywords(
            text=text,
            language=language
        )

        return ResponseType[KeywordExtractionDataClass](
            original_response=response,
            standardized_response=normalize_keywords(response)
        )
