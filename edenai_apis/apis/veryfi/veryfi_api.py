import mimetypes
import pathlib
import uuid
from http import HTTPStatus
from typing import Dict, Literal

import boto3
import requests
from requests.exceptions import JSONDecodeError

from edenai_apis.apis.veryfi.veryfi_ocr_normalizer import (
    veryfi_bank_check_parser,
    veryfi_financial_parser,
    veryfi_invoice_parser,
    veryfi_receipt_parser,
)
from edenai_apis.features.ocr.bank_check_parsing import BankCheckParsingDataClass
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import load_key
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class VeryfiApi(ProviderInterface, OcrInterface):
    provider_name = "veryfi"

    def __init__(self, api_keys: Dict = {}):
        self.url = "https://api.veryfi.com/api/v8/partner"

        self.api_settings = load_key(
            provider_name=self.provider_name, api_keys=api_keys
        )
        self.client_id = self.api_settings["client_id"]
        self.client_secret = self.api_settings["client_secret"]
        self.authorization = self.api_settings["Authorization"]

        # settings for veryfi partners, upload documents to veryfi's bucket for processing
        self.partner_iam_api_key_id = self.api_settings.get("optional_iam_api_key_id")
        self.partner_iam_api_key_secret = self.api_settings.get(
            "optional_iam_api_key_secret"
        )
        self.partner_bucket_name = self.api_settings.get("optional_bucket_name")
        self.partner_upload_folder = self.api_settings.get("optional_upload_folder")

        self.headers = {
            "Accept": "application/json",
            "CLIENT-ID": self.client_id,
            "Authorization": self.authorization,
        }

    def _process_document(
        self, file: str, document_type: Literal["documents", "checks"] = "documents"
    ):
        is_partner = all(
            [
                self.partner_iam_api_key_id,
                self.partner_iam_api_key_secret,
                self.partner_bucket_name,
                self.partner_upload_folder,
            ]
        )
        if is_partner:
            response = self._process_document_through_bucket(file, document_type)
        else:
            response = self._process_document_directly(file, document_type)

        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(message="Internal Server Error", code=500)

        if response.status_code != HTTPStatus.CREATED:
            error_message = original_response.get("message") or original_response.get(
                "error"
            )
            raise ProviderException(message=error_message, code=response.status_code)

        return original_response

    def _process_document_through_bucket(self, filename: str, document_type: str):
        """Upload file to veryfi bucket then process it"""
        client = boto3.client(
            "s3",
            aws_access_key_id=self.partner_iam_api_key_id,
            aws_secret_access_key=self.partner_iam_api_key_secret,
        )

        random_filename = (
            f"{document_type}_{uuid.uuid4()}{pathlib.Path(filename).suffix}"
        )

        client.upload_file(
            filename,
            self.partner_bucket_name,
            f"{self.partner_upload_folder}/{random_filename}",
        )

        return requests.request(
            method="POST",
            url=f"{self.url}/{document_type}",
            headers=self.headers,
            json={
                "bucket": self.partner_bucket_name,
                "package_path": f"{self.partner_upload_folder}/{random_filename}",
            },
        )

    def _process_document_directly(self, file: str, document_type: str):
        """Send file directly to veryfi for processing"""
        with open(file, "rb") as file_:
            payload = {"file_name": f"test-{uuid.uuid4()}"}

            files = {"file": ("file", file_, mimetypes.guess_type(file_.name)[0])}

            return requests.request(
                method="POST",
                url=f"{self.url}/{document_type}",
                headers=self.headers,
                data=payload,
                files=files,
            )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        original_response = self._process_document(file)

        standardized_response = veryfi_invoice_parser(original_response)
        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        original_response = self._process_document(file)

        standardized_response = veryfi_receipt_parser(original_response)

        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__bank_check_parsing(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[BankCheckParsingDataClass]:
        original_response = self._process_document(file, document_type="checks")

        standardized_response = veryfi_bank_check_parser(original_response)

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
        original_response = self._process_document(file)

        standardized_response = veryfi_financial_parser(original_response)

        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
