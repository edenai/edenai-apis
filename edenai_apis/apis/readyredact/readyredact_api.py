from typing import Dict

import requests

from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import AnonymizationDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.types import ResponseType


class ReadyRedactApi(ProviderInterface, OcrInterface):
    provider_name = "readyredact"
    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = api_settings["api_key"]
        self.email = api_settings["email"]
        self.url_put_file = "https://api.readyredact.com/v1/document/put-file"

    def ocr__anonymization(
        self, file: str, file_url: str = ""
    ) -> ResponseType[AnonymizationDataClass]:
        header = {
            "api_key": self.api_key
        }
        file_ = open(file, "rb")
        files = {"file[]": file_}
        payload = {
            "email": self.email
        }
        response = requests.post(self.url_put_file, header=header, data=payload, files=files)
        
