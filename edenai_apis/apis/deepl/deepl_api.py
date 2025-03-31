import base64
import http.client
import json
import mimetypes
from io import BytesIO
from time import sleep
from typing import Dict, Optional

import requests

from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.translation.automatic_translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features.translation.document_translation.document_translation_dataclass import (
    DocumentTranslationDataClass,
)
from edenai_apis.features.translation.translation_interface import TranslationInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3, USER_PROCESS


class DeeplApi(ProviderInterface, TranslationInterface):
    provider_name = "deepl"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.deepl.com/v2/"
        self.header = {
            "authorization": f"DeepL-Auth-Key {self.api_key}",
        }

    def translation__automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}translate"

        data = {
            "text": text,
            "source_lang": source_language,
            "target_lang": target_language,
        }

        response = requests.request("POST", url, headers=self.header, data=data)

        if response.status_code >= 500:
            raise ProviderException(message=response.text, code=response.status_code)

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message=response.text, code=response.status_code
            ) from exc

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        standardized_response = AutomaticTranslationDataClass(
            text=original_response["translations"][0]["text"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def translation__document_translation(
        self,
        file: str,
        file_type: str,
        source_language: str,
        target_language: str,
        file_url: str = "",
        **kwargs,
    ) -> ResponseType[DocumentTranslationDataClass]:
        mimetype = mimetypes.guess_type(file)[0]
        extension = mimetypes.guess_extension(mimetype)

        with open(file, "rb") as file_:
            files = {
                "file": file_,
            }

            data = {"target_lang": target_language, "source_lang": source_language}

            try:
                response = requests.post(
                    f"{self.url}document", headers=self.header, data=data, files=files
                )
            except:
                raise ProviderException(
                    "Something went wrong when performing document translation!!", 500
                )
            if response.status_code >= 400:
                raise ProviderException(
                    message=http.client.responses[response.status_code],
                    code=response.status_code,
                )
            try:
                original_response = response.json()
            except json.JSONDecodeError:
                raise ProviderException("Internal server error", 500)

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        document_id, document_key = response.json().values()

        doc_key = {"document_key": document_key}

        response_status = requests.post(
            f"{self.url}document/{document_id}", headers=self.header, data=doc_key
        ).json()
        try:
            while response_status["status"] != "done":
                response_status = requests.post(
                    f"{self.url}document/{document_id}",
                    headers=self.header,
                    data=doc_key,
                ).json()
                if response_status["status"] == "error":
                    raise ProviderException(response_status["error_message"])
                sleep(0.5)
        except KeyError as exc:
            raise ProviderException("Internal server error", 500) from exc

        response = requests.post(
            f"{self.url}document/{document_id}/result",
            headers=self.header,
            data=doc_key,
        )

        b64_file = base64.b64encode(response.content)
        resource_url = upload_file_bytes_to_s3(
            BytesIO(response.content), extension, USER_PROCESS
        )

        std_resp = DocumentTranslationDataClass(
            file=b64_file, document_resource_url=resource_url
        )
        return ResponseType[DocumentTranslationDataClass](
            original_response=response.content, standardized_response=std_resp
        )
