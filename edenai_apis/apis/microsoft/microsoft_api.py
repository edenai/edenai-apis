from typing import Dict
from edenai_apis.apis.microsoft.microsoft_audio_api import MicrosoftAudioApi
from edenai_apis.apis.microsoft.microsoft_helpers import (
    get_microsoft_headers,
    get_microsoft_urls,
)
from edenai_apis.apis.microsoft.microsoft_image_api import MicrosoftImageApi
from edenai_apis.apis.microsoft.microsoft_ocr_api import MicrosoftOcrApi
from edenai_apis.apis.microsoft.microsoft_text_api import MicrosoftTextApi
from edenai_apis.apis.microsoft.microsoft_translation_api import MicrosoftTranslationApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


class MicrosoftApi(
    ProviderInterface,
    MicrosoftImageApi,
    MicrosoftTextApi,
    MicrosoftTranslationApi,
    MicrosoftOcrApi,
    MicrosoftAudioApi,
):
    provider_name = "microsoft"

    def __init__(self, user=None, api_keys: Dict = {}):
        super().__init__()

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.headers = get_microsoft_headers()
        self.url = get_microsoft_urls()
        self.user = user
