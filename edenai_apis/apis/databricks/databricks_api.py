import os
from typing import Dict

from edenai_apis.apis.databricks.databricks_llm_api import DatabricksLLMApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


class DatabricksApi(ProviderInterface, DatabricksLLMApi):
    provider_name = "databricks"

    def __init__(self, api_keys: Dict = {}):

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        provider_config = {
            "api_base": self.api_settings["host"],
            "api_key": self.api_settings["api_key"],
        }

        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config=provider_config
        )
