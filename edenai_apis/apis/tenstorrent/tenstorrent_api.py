from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.apis.tenstorrent.tenstorrent_text_api import \
    TenstorrentTextApi


class TenstorrentAPI(
    ProviderInterface,
    TenstorrentTextApi,
):
    provider_name = "tenstorrent"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, provider_name=self.provider_name)
