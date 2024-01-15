import base64
import json
import mimetypes
import uuid
from typing import Dict
from io import BytesIO

import requests

from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr.anonymization_async.anonymization_async_dataclass import (
    AnonymizationAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from apis.amazon.helpers import check_webhook_result
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3, USER_PROCESS


class PrivateaiApi(ProviderInterface, OcrInterface):
    provider_name = "privateai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.private-ai.com/deid/"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings.get("webhook_token")

    def ocr__anonymization_async__launch_job(
        self, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        data_job_id = {}
        file_ = open(file, "rb")
        file_data = base64.b64encode(file_.read())
        file_data = file_data.decode("ascii")
        mimetype = mimetypes.guess_type(file)[0]
        extension = mimetypes.guess_extension(mimetype)
        file_.close()
        data = {
            "file": {
                "data": file_data,  # base64 converted file
                "content_type": mimetype,
            },
            "entity_detection": {
                "accuracy": "high",
                "return_entity": True,
            },
        }
        response = requests.post(
            url=self.url + "v3/process/files/base64",
            data=json.dumps(data),
            headers=self.headers,
        )
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        original_response = response.json()
        original_response["extension"] = extension
        job_id = "document_anonymization_privateai" + str(uuid.uuid4())
        data_job_id[job_id] = original_response
        requests.post(
            url=f"https://webhook.site/{self.webhook_token}",
            data=json.dumps(data_job_id),
            headers={"content-type": "application/json"},
        )

        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def ocr__anonymization_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AnonymizationAsyncDataClass]:
        wehbook_result, response_status = check_webhook_result(
            provider_job_id, self.webhook_settings
        )

        if response_status != 200:
            raise ProviderException(wehbook_result, code=response_status)

        result_object = (
            next(
                filter(
                    lambda response: provider_job_id in response["content"],
                    wehbook_result,
                ),
                None,
            )
            if wehbook_result
            else None
        )

        if not result_object or not result_object.get("content"):
            raise ProviderException("Provider returned an empty response")

        try:
            original_response = json.loads(result_object["content"]).get(
                provider_job_id, None
            )
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")

        if original_response is None:
            return AsyncPendingResponseType[AnonymizationAsyncDataClass](
                provider_job_id=provider_job_id
            )
        # Extract the B64 redacted document
        redacted_document = original_response["processed_file"]
        document_extension = original_response["extension"]

        content_bytes = base64.b64decode(redacted_document)
        resource_url = upload_file_bytes_to_s3(
            BytesIO(content_bytes), document_extension, USER_PROCESS
        )
        return AsyncResponseType[AnonymizationAsyncDataClass](
            original_response=original_response,
            standardized_response=AnonymizationAsyncDataClass(
                document=redacted_document, document_url=resource_url
            ),
            provider_job_id=provider_job_id,
        )
