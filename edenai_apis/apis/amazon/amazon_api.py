from edenai_apis.apis.amazon.amazon_audio_api import AmazonAudioApi
from edenai_apis.apis.amazon.amazon_image_api import AmazonImageApi
from edenai_apis.apis.amazon.amazon_ocr_api import AmazonOcrApi
from edenai_apis.apis.amazon.amazon_text_api import AmazonTextApi
from edenai_apis.apis.amazon.amazon_translation_api import AmazonTranslationApi
from edenai_apis.apis.amazon.amazon_video_api import AmazonVideoApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from .config import clients, storage_clients


class AmazonApi(
    ProviderInterface,
    AmazonOcrApi,
    AmazonAudioApi,
    AmazonImageApi,
    AmazonTextApi,
    AmazonTranslationApi,
    AmazonVideoApi,
):
    provider_name = "amazon"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
        self.clients = clients(self.api_settings)
        self.storage_clients = storage_clients(self.api_settings)
