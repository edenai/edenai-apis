from typing import Dict

import requests

from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import AnonymizationAsyncDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType, AsyncBaseResponseType


class ReadyRedactApi(ProviderInterface, OcrInterface):
    provider_name = "readyredact"
    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = api_settings["api_key"]
        self.email = api_settings["email"]
        self.url_put_file = "https://api.readyredact.com/v1/document/put-file"
        self.url_get_file = "https://api.readyredact.com/v1/document/get-file"
        self.webhook_url = f"https://webhook.site/55722cad-1d14-42da-b1d2-951cc16c2d4d"


    def ocr__anonymization_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncBaseResponseType:

        file_ = open(file, "rb")
        files = [
            ('file[]', (file, file_, 'application/pdf'))
        ]
        payload = {
            "email": self.email
        }
        headers = {
            'Accept': 'application/json'
        }
        params = {
            "api_key": self.api_key
        }
        response = requests.post(url=self.url_put_file, params=params, data=payload, files=files, headers=headers)
        raise ProviderException(response)
        original_response = response.json()
        raise ProviderException(original_response)
        return AsyncBaseResponseType(provider_job_id="12345")

    def ocr__anonymization_async__get_job_result(
            self, provider_job_id: str
    ) -> AsyncBaseResponseType[AnonymizationAsyncDataClass]:
        original_response = {}
        response = "file"
        return AsyncBaseResponseType[AnonymizationAsyncDataClass](
            original_response=original_response,
            standardized_response=AnonymizationAsyncDataClass(response),
            provider_job_id=provider_job_id
        )