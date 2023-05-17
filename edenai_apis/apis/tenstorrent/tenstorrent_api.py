from typing import Dict
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.apis.tenstorrent.tenstorrent_text_api import TenstorrentTextApi


class TenstorrentApi(
    ProviderInterface,
    TenstorrentTextApi,
):
    provider_name = "tenstorrent"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys = api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://sentiment-analysis--eden-ai.workload.tenstorrent.com"
        self.headers = {
            "accept": "application/json",
            "authorization": self.api_key,
            "content-type": "application/json",
            "Tenstorrent-Version": "2023-05-15",
        }
