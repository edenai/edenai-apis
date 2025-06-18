import base64
import json
import mimetypes
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from apis.amazon.helpers import check_webhook_result

from edenai_apis.features import OcrInterface, ProviderInterface, TextInterface
from edenai_apis.features.ocr.anonymization_async.anonymization_async_dataclass import (
    AnonymizationAsyncDataClass,
)
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
    AnonymizationEntity,
)
from edenai_apis.features.text.anonymization.category import CategoryType
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.parsing import extract
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class PrivateaiApi(ProviderInterface, OcrInterface, TextInterface):
    provider_name = "privateai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.private-ai.com/cloud/"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings.get("webhook_token")

    def ocr__anonymization_async__launch_job(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        if provider_params is None:
            provider_params = {}
        entity_detection_params = provider_params.get("entity_detection", {}) or {}
        processed_text_params = provider_params.get("processed_text", {}) or {}
        is_multilingual = provider_params.get("multilingual", False) or False

        with open(file, "rb") as file_:
            file_data = base64.b64encode(file_.read())
            file_data = file_data.decode("ascii")
            mimetype = mimetypes.guess_type(file)[0]
            extension = mimetypes.guess_extension(mimetype)
        data = {
            "file": {
                "data": file_data,  # base64 converted file
                "content_type": mimetype,
            },
            "entity_detection": {
                "accuracy": "high" if not is_multilingual else "high_multilingual",
                "return_entity": True,
                "enable_non_max_suppression": True,
                **entity_detection_params,
            },
        }
        if processed_text_params:
            data["processed_text"] = processed_text_params

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
            provider_job_id=job_id,
        )

    def text__anonymization(
        self,
        text: str,
        language: str,
        model: Optional[str] = None,
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[AnonymizationDataClass]:
        if provider_params is None:
            provider_params = {}
        entity_detection_params = provider_params.get("entity_detection", {}) or {}
        processed_text_params = provider_params.get("processed_text", {}) or {}
        payload = {
            "text": [text],
            "entity_detection": {
                "accuracy": "high",
                "return_entity": True,
                **entity_detection_params,
            },
        }
        if processed_text_params:
            payload["processed_text"] = processed_text_params
        response = requests.post(
            url=self.url + "v3/process/text", json=payload, headers=self.headers
        )

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(message="Internal server error", code=500) from exc

        entities: List[AnonymizationEntity] = []
        assemble_text = True
        new_text = text
        if processed_text_params.get(
            "type", ""
        ) == "MASK" and processed_text_params.get("mask_character"):
            assemble_text = False
            new_text = original_response[0].get("processed_text", "") or ""
        for entity in original_response[0].get("entities", []):
            start_index = extract(entity, ["location", "stt_idx"])
            end_index = extract(entity, ["location", "end_idx"])
            original_label = entity.get("best_label")
            confidence = extract(entity, ["labels", original_label])
            classification = CategoryType.choose_category_subcategory(original_label)
            replacement = "*" * (end_index - start_index)
            if assemble_text:
                new_text = new_text[:start_index] + replacement + new_text[end_index:]
            entities.append(
                AnonymizationEntity(
                    offset=start_index,
                    length=end_index - start_index,
                    category=classification["category"],
                    subcategory=classification["subcategory"],
                    original_label=original_label,
                    content=entity.get("text"),
                    confidence_score=confidence,
                )
            )

        standardized_response = AnonymizationDataClass(
            result=new_text, entities=entities
        )
        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
