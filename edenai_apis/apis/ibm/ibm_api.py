from typing import Dict

from edenai_apis.apis.ibm.ibm_text_api import IbmTextApi
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from .config import ibm_clients


class IbmApi(ProviderInterface, IbmTextApi):
    provider_name = "ibm"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, "ibm", api_keys=api_keys
        )
        self.clients = ibm_clients(self.api_settings)
