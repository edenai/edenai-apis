import os
from io import BufferedReader
from json import JSONDecodeError
from typing import Dict
import aiofiles

import requests

from edenai_apis.utils.http_client import async_client, OCR_TIMEOUT

from edenai_apis.apis.eagledoc.eagledoc_ocr_normalizer import (
    eagledoc_financial_parser,
    eagledoc_invoice_parser,
    eagledoc_receipt_parser,
)
from edenai_apis.features import OcrInterface, ProviderInterface
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.loaders.loaders import ProviderDataEnum, load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.file_handling import FileHandler
from edenai_apis.utils.types import ResponseType


class EagledocApi(ProviderInterface, OcrInterface):
    provider_name = "eagledoc"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["subscription_key"]
        self.url = "https://de.eagle-doc.com/api"

        self.params = {}

        self.headers = {
            "api-key": self.api_key,
        }

    def _make_post_request(self, file: BufferedReader, endpoint: str = ""):
        files = {
            "file": file,
        }

        response = requests.post(
            url=self.url + endpoint,
            headers=self.headers,
            files=files,
            params=self.params,
        )

        try:
            original_response = response.json()
        except JSONDecodeError as exc:
            raise ProviderException(message="Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

    async def _amake_post_request(
        self, file_content: bytes, filename: str, endpoint: str = ""
    ):
        files = {
            "file": (filename, file_content),
        }

        async with async_client(OCR_TIMEOUT) as client:
            response = await client.post(
                url=self.url + endpoint,
                headers=self.headers,
                files=files,
                params=self.params,
            )

        try:
            original_response = response.json()
        except JSONDecodeError as exc:
            raise ProviderException(message="Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:

        with open(file, "rb") as file_:
            original_response = self._make_post_request(
                file_, endpoint="/finance/v1/processing"
            )

        standardize_response = eagledoc_financial_parser(original_response)

        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )

    async def ocr__afinancial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        file_handler = FileHandler()
        file_wrapper = None

        try:
            if file:
                async with aiofiles.open(file, "rb") as file_:
                    file_content = await file_.read()
                filename = os.path.basename(file)
            elif file_url:
                file_wrapper = await file_handler.download_file(file_url)
                file_content = await file_wrapper.get_bytes()
                extension = file_wrapper.file_info.file_extension or "pdf"
                filename = f"document.{extension}"
            else:
                raise ProviderException(
                    "Either file or file_url must be provided", code=400
                )

            original_response = await self._amake_post_request(
                file_content, filename=filename, endpoint="/finance/v1/processing"
            )

            standardize_response = eagledoc_financial_parser(original_response)
            return ResponseType[FinancialParserDataClass](
                original_response=original_response,
                standardized_response=standardize_response,
            )
        finally:
            if file_wrapper:
                file_wrapper.close_file()

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        with open(file, "rb") as file_:
            original_response = self._make_post_request(
                file_, endpoint="/invoice/v1/processing"
            )

        standardize_response = eagledoc_invoice_parser(original_response)

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        with open(file, "rb") as file_:
            original_response = self._make_post_request(
                file_, endpoint="/receipt/v3/processing"
            )

        standardize_response = eagledoc_receipt_parser(original_response)
        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )
