from typing import Dict

from edenai_apis.apis.tenstorrent.tenstorrent_text_api import TenstorrentTextApi
from edenai_apis.apis.tenstorrent.tenstorrent_llm_api import TenstorrentLLMApi

from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from openai import OpenAI
from edenai_apis.llmengine.llm_engine import LLMEngine, StdLLMEngine


class TenstorrentApi(
    ProviderInterface,
    TenstorrentTextApi,
    TenstorrentLLMApi,
):
    provider_name = "tenstorrent"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "accept": "application/json",
            "authorization": self.api_key,
            "content-type": "application/json",
            "Tenstorrent-Version": "2023-06-26",
        }
        self.url = "https://chat-and-generation--eden-ai.workload.tenstorrent.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
        self.llm_client = LLMEngine(provider_name=self.provider_name,provider_config={"api_key": self.api_key})
        self.std_llm_client = StdLLMEngine(provider_config={"api_key": self.api_key})
        self.moderation_flag = True
