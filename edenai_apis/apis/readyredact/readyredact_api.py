import base64
import json
from io import BytesIO
from typing import Dict

import magic
import requests

from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import AnonymizationAsyncDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncResponseType,
    AsyncPendingResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class ReadyRedactApi(ProviderInterface, OcrInterface):
    provider_name = "readyredact"

    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.api_key = api_settings["api_key"]
        self.email = api_settings["email"]
        self.url_put_file = "https://api.readyredact.com/v1/document/put-file"
        self.url_get_file = (
            f"https://api.readyredact.com/v1/document/get-file?api_key={self.api_key}"
        )

    def ocr__anonymization_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:

        with open(file, "rb") as file_:
            files = [("file[]", (file, file_, "application/pdf"))]
            payload = {"email": self.email}
            headers = {"Accept": "application/json"}
            params = {"api_key": self.api_key}
            response = requests.post(
                url=self.url_put_file,
                params=params,
                data=payload,
                files=files,
                headers=headers,
            )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)
        try:
            original_response_put = response.json()
            document_id = (
                original_response_put[0].get("details", {}).get("document_id", "")
            )
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        return AsyncLaunchJobResponseType(provider_job_id=document_id)

    def ocr__anonymization_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AnonymizationAsyncDataClass]:
        payload = json.dumps(
            {"email": self.email, "document_id": provider_job_id, "pdf_download": False}
        )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = requests.request(
            "GET", self.url_get_file, headers=headers, data=payload
        )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)
        try:
            original_response_get = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        status = original_response_get.get("status", False)
        if status:
            document = original_response_get["document"]
            document_binary = base64.b64decode(document)
            document_content = BytesIO(document_binary)
            file_type = magic.from_buffer(document_content.read(), mime=True)
            document_content.seek(0)
            file_extension = file_type.split("/")[1]
            document_url = upload_file_bytes_to_s3(
                document_content, f".{file_extension}", USER_PROCESS
            )
            return AsyncResponseType[AnonymizationAsyncDataClass](
                original_response=original_response_get,
                standardized_response=AnonymizationAsyncDataClass(
                    document=document, document_url=document_url
                ),
                provider_job_id=provider_job_id,
            )
        else:
            return AsyncPendingResponseType[AnonymizationAsyncDataClass](
                provider_job_id=provider_job_id
            )
