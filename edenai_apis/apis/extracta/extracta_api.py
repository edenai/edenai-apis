import base64
import json
import mimetypes
from typing import List, Dict, Union

import requests

from edenai_apis.apis.extracta.extracta_ocr_normalizer import (
    extracta_resume_parser,
    extracta_bank_check_parsing,
    extracta_financial_parser,
)
from edenai_apis.features.ocr.resume_parser import ResumeParserDataClass
from edenai_apis.features.ocr.bank_check_parsing import BankCheckParsingDataClass
from edenai_apis.features.ocr.financial_parser import FinancialParserDataClass
from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import (
    CustomDocumentParsingAsyncBoundingBox,
    CustomDocumentParsingAsyncDataClass,
    CustomDocumentParsingAsyncItem,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)


class ExtractaApi(
    ProviderInterface,
    OcrInterface,
):
    provider_name = "extracta"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, "extracta", api_keys=api_keys
        )

        self.api_key = self.api_settings["api_key"]
        self.url = self.api_settings["url"]
        self.uploadFileRoute = self.api_settings["uploadFileRoute"]
        self.getResultRoute = self.api_settings["getResultRoute"]
        self.processFileRoute = self.api_settings["processFileRoute"]

    def ocr__custom_document_parsing_async__launch_job(
        self,
        file: str,
        queries: List[Dict[str, Union[str, str]]],
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        isUrl = False

        # Check if file_url is provided
        if file_url:
            image_source = file_url
            isUrl = True
        else:
            # Open the file and read its contents
            try:
                with open(file, "rb") as f_stream:
                    image_as_base64 = (
                        f"data:{mimetypes.guess_type(file)[0]};base64,"
                        + base64.b64encode(f_stream.read()).decode()
                    )
                image_source = image_as_base64
            except FileNotFoundError:
                raise ProviderException("Error: The file was not found.")
            except IOError:
                raise ProviderException(
                    "Error: An I/O error occurred while handling the file."
                )

        # check if queries are provided
        if not queries:
            raise ProviderException("Error: No queries provided.")

        # make formatted fields
        formatted_fields = [
            {"key": query.get("query"), "Pages": query.get("pages").split(",")}
            for query in queries
        ]

        # make payload
        payload = json.dumps(
            {
                "extractionDetails": {
                    "name": "Eden.ai - Extraction",
                    "language": "English",
                    "fields": formatted_fields,
                },
                "file": image_source,
                "isUrl": isUrl,
            }
        )

        # make headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # call api
        response = requests.post(
            url=self.url + self.uploadFileRoute, headers=headers, data=payload
        )

        # check for error
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        # successful response
        original_response = response.json()

        # extract job id
        job_id = original_response.get("job_id", None)

        # check for job id
        if not job_id:
            raise ProviderException("Error: Job ID not found in response.")

        # return response
        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def ocr__custom_document_parsing_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[CustomDocumentParsingAsyncDataClass]:
        # make payload
        payload = json.dumps(
            {
                "job_id": provider_job_id,
            }
        )

        # make headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # call api
        response = requests.post(
            url=self.url + self.getResultRoute, headers=headers, data=payload
        )

        # check for error
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        # successful response
        original_response = response.json()

        # get status from request
        status = original_response.get("status", None)

        # check for status
        if not status:
            raise ProviderException("Error: Status not found in response.")

        if status == "not_found":
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                code=response.status_code,
            )

        if status == "error":
            raise ProviderException(
                "Error: An error occurred while processing the request."
            )

        if status == "pending":
            return AsyncPendingResponseType[CustomDocumentParsingAsyncDataClass](
                provider_job_id=provider_job_id
            )

        if status != "success":
            raise ProviderException("Error: Unknown status.")

        # get job id from request and comapre it
        if original_response.get("job_id", None) != provider_job_id:
            raise ProviderException(
                "Error: Job ID from request does not match the job id from response."
            )

        # get extraction details
        extraction_details = original_response.get("extractionDetails", None)
        if not extraction_details:
            raise ProviderException("Error: Extraction details not found in response.")

        # prepare the standardized response
        items = []
        for key, value in extraction_details.items():
            item = CustomDocumentParsingAsyncItem(
                confidence=1,
                value=str(value),
                query=key,
                page=0,
                bounding_box=CustomDocumentParsingAsyncBoundingBox(
                    left=0,
                    top=0,
                    width=0,
                    height=0,
                ),
            )
            items.append(item)

        standardized_response = CustomDocumentParsingAsyncDataClass(items=items)

        return AsyncResponseType[CustomDocumentParsingAsyncDataClass](
            original_response=extraction_details,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )

    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:
        isUrl = False

        # Check if file_url is provided
        if file_url:
            image_source = file_url
            isUrl = True
        else:
            # Open the file and read its contents
            try:
                with open(file, "rb") as f_stream:
                    image_as_base64 = (
                        f"data:{mimetypes.guess_type(file)[0]};base64,"
                        + base64.b64encode(f_stream.read()).decode()
                    )
                image_source = image_as_base64
            except FileNotFoundError:
                raise ProviderException("Error: The file was not found.")
            except IOError:
                raise ProviderException(
                    "Error: An I/O error occurred while handling the file."
                )

        # make payload
        payload = json.dumps(
            {
                "extractionDetails": {
                    "name": "Eden.ai - Extraction",
                    "language": "English",
                    "documentId": "resume_parser",
                },
                "file": image_source,
                "isUrl": isUrl,
            }
        )

        # make headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # call api
        response = requests.post(
            url=self.url + self.processFileRoute, headers=headers, data=payload
        )

        # check for error
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        # successful response
        original_response = response.json()

        standardized_response = extracta_resume_parser(original_response)
        return ResponseType[ResumeParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__bank_check_parsing(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[BankCheckParsingDataClass]:
        isUrl = False

        # Check if file_url is provided
        if file_url:
            image_source = file_url
            isUrl = True
        else:
            # Open the file and read its contents
            try:
                with open(file, "rb") as f_stream:
                    image_as_base64 = (
                        f"data:{mimetypes.guess_type(file)[0]};base64,"
                        + base64.b64encode(f_stream.read()).decode()
                    )
                image_source = image_as_base64
            except FileNotFoundError:
                raise ProviderException("Error: The file was not found.")
            except IOError:
                raise ProviderException(
                    "Error: An I/O error occurred while handling the file."
                )

        # make payload
        payload = json.dumps(
            {
                "extractionDetails": {
                    "name": "Eden.ai - Extraction",
                    "language": "English",
                    "documentId": "bank_check_parsing",
                },
                "file": image_source,
                "isUrl": isUrl,
            }
        )

        # make headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # call api
        response = requests.post(
            url=self.url + self.processFileRoute, headers=headers, data=payload
        )

        # check for error
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        # successful response
        original_response = response.json()

        standardized_response = extracta_bank_check_parsing(original_response)
        return ResponseType[BankCheckParsingDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        isUrl = False

        # Check if file_url is provided
        if file_url:
            image_source = file_url
            isUrl = True
        else:
            # Open the file and read its contents
            try:
                with open(file, "rb") as f_stream:
                    image_as_base64 = (
                        f"data:{mimetypes.guess_type(file)[0]};base64,"
                        + base64.b64encode(f_stream.read()).decode()
                    )
                image_source = image_as_base64
            except FileNotFoundError:
                raise ProviderException("Error: The file was not found.")
            except IOError:
                raise ProviderException(
                    "Error: An I/O error occurred while handling the file."
                )

        # make payload
        payload = json.dumps(
            {
                "extractionDetails": {
                    "name": "Eden.ai - Extraction",
                    "language": "English",
                    "documentId": "financial_parser",
                    "documentType": document_type,
                    "documentLanguage": language,
                },
                "file": image_source,
                "isUrl": isUrl,
            }
        )

        # make headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # call api
        response = requests.post(
            url=self.url + self.processFileRoute, headers=headers, data=payload
        )

        # check for error
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        # successful response
        original_response = response.json()

        standardized_response = extracta_financial_parser(original_response)
        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
