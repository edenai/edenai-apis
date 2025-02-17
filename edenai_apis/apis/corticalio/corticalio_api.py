from typing import Dict, Optional

from edenai_apis.apis.corticalio.client import CorticalClient
from edenai_apis.apis.corticalio.helpers import normalize_keywords
from edenai_apis.features import TextInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.text import KeywordExtractionDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class CorticalioApi(ProviderInterface, TextInterface):
    provider_name = "corticalio"

    def __init__(self, api_keys: Dict = None):
        api_keys = api_keys or {}
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.client = CorticalClient(self.api_settings)

    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        Extract Keywords from a given text

        Args:
            text (str): text to analyze, required and must not be empty or blank
            language (str): text's language code in ISO format,
                            will be auto-detected if not specified.

        Raises:
            ProviderException: if the specified or auto-detected language is not supported, or
                               some error occurred while processing request.
        """
        response = self.client.extract_keywords(text=text, language=language)

        return ResponseType[KeywordExtractionDataClass](
            original_response=response,
            standardized_response=normalize_keywords(response),
        )
