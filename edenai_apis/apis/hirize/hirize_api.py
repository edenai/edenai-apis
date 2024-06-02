from edenai_apis.features.provider.provider_interface import ProviderInterface
from typing import Dict, Any
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import ResumeParserDataClass
import random
import requests
import json
from http import HTTPStatus
from io import BufferedReader
from json import JSONDecodeError
from edenai_apis.utils.exception import ProviderException
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
        self.base_url = "https://connect.hirize.hr/api/public/"

        self.url_security_param = "/?api_key=" + self.api_key
        self.headers = {
            'Content-Type': 'application/json'
        }

    def orc__resume_parser(self, payload: str, file_name: str) -> ResponseType[ResumeParserDataClass]:

        url = self.base_url + "parser" + self.url_security_param
        dumpData = json.dumps({
            "payload": payload,
            "file_name": file_name
        })

        result = Client(api_keys=self.api_key,
                        header=self.headers,
                        data=dumpData,
                        url=url
                    ).ocr_resume_parser()

        return result

