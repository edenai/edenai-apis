from typing import Dict

from edenai_apis.apis.amazon.amazon_llm_api import AmazonLLMApi
from edenai_apis.apis.amazon.amazon_audio_api import AmazonAudioApi
from edenai_apis.apis.amazon.amazon_image_api import AmazonImageApi
from edenai_apis.apis.amazon.amazon_ocr_api import AmazonOcrApi
from edenai_apis.apis.amazon.amazon_text_api import AmazonTextApi
from edenai_apis.apis.amazon.amazon_translation_api import AmazonTranslationApi
from edenai_apis.apis.amazon.amazon_video_api import AmazonVideoApi
from edenai_apis.apis.amazon.amazon_multimodal_api import AmazonMultimodalApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.llmengine.llm_engine import LLMEngine
from .config import clients, storage_clients


class AmazonApi(
    ProviderInterface,
    AmazonOcrApi,
    AmazonAudioApi,
    AmazonImageApi,
    AmazonTextApi,
    AmazonTranslationApi,
    AmazonVideoApi,
    AmazonMultimodalApi,
    AmazonLLMApi,
):
    provider_name = "amazon"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, "amazon", api_keys=api_keys
        )
        self.clients = clients(self.api_settings)
        self.storage_clients = storage_clients(self.api_settings)
        self.llm_client = LLMEngine(
            provider_name="bedrock",
            provider_config={
                "aws_access_key_id": self.api_settings["aws_access_key_id"],
                "aws_secret_access_key": self.api_settings["aws_secret_access_key"],
                "aws_region_name": "us-east-1",
            },
        )
