import os
from typing import Dict

from edenai_apis.apis.databricks.databricks_llm_api import DatabricksLLMApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

# Flag to ensure environment variables are only set once at module load
_DATABRICKS_ENV_INITIALIZED = False


class DatabricksApi(ProviderInterface, DatabricksLLMApi):
    provider_name = "databricks"

    def __init__(self, api_keys: Dict = {}):
        global _DATABRICKS_ENV_INITIALIZED

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        if "client_id" in self.api_settings and "client_secret" in self.api_settings:
            # OAuth M2M authentication (recommended for production)
            # LiteLLM reads credentials from environment variables
            # Since all users share the same Databricks account, set env vars once
            if not _DATABRICKS_ENV_INITIALIZED:
                os.environ["DATABRICKS_CLIENT_ID"] = self.api_settings["client_id"]
                os.environ["DATABRICKS_CLIENT_SECRET"] = self.api_settings[
                    "client_secret"
                ]
                os.environ["DATABRICKS_API_BASE"] = self.api_settings["host"]

                _DATABRICKS_ENV_INITIALIZED = True

        else:
            raise ValueError(
                "Databricks authentication requires either (client_id + client_secret) for OAuth"
            )

        self.llm_client = LLMEngine(provider_name=self.provider_name)
