from typing import Dict
from edenai_apis.apis.xai.xai_multimodal_api import XAiMultimodalApi
from edenai_apis.apis.xai.xai_text_api import XAiTextApi
from edenai_apis.apis.xai.xai_translation_api import XAiTranslationApi
from edenai_apis.apis.xai.xai_llm_api import XAiLLMApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.llmengine.llm_engine import LLMEngine


class XAiApi(
    ProviderInterface, XAiTextApi, XAiTranslationApi, XAiMultimodalApi, XAiLLMApi
):
    provider_name = "xai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )
