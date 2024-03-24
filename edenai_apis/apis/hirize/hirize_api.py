from edenai_apis.features.provider.provider_interface import ProviderInterface
from typing import Dict
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import ResumeParserDataClass
import random
import requests
import json
from .client import Client

class HirizeApi(ProviderInterface):

    provider_name: str = "hirize"
    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        if isinstance(self.api_settings, list):
            chosen_api_setting = random.choice(self.api_settings)
        else:
            chosen_api_setting = self.api_settings

        self.api_key = chosen_api_setting["api_key"]
        self.url = "https://connect.hirize.hr/api/public/?api_key=" + self.api_key
        self.headers = {
                            'Content-Type': 'application/json'
                       }

    def resume_parser(self, payload: str, file_name: str) -> ResponseType[ResumeParserDataClass]:

            dumpData = json.dumps({
                "payload": payload,
                "file_name": file_name
            })

            hirize_response = requests.request("POST", self.url, headers=self.headers, data=dumpData)

            return ResponseType[ResumeParserDataClass](
                original_response=hirize_response.json()
            )