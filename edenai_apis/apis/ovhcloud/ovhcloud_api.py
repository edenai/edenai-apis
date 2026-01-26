from typing import Dict

from edenai_apis.apis.ovhcloud.ovhcloud_llm_api import OvhCloudLLMApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


class OvhCloudApi(ProviderInterface, OvhCloudLLMApi):
    provider_name = "ovhcloud"

    def __init__(self, api_keys: dict | None = None):
        api_keys = api_keys or {}
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )
