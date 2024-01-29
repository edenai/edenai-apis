from io import BufferedReader
from json import JSONDecodeError
from typing import Dict

import requests

from edenai_apis.features import OcrInterface, ProviderInterface
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.loaders.loaders import ProviderDataEnum, load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.eagledoc.eagledoc_ocr_normalizer import (
    eagledoc_invoice_parser,
    eagledoc_receipt_parser
)

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
            url=self.url + endpoint, headers=self.headers, files=files, params=self.params
        )

        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(
                message="Internal Server Error", code=500
            ) from JSONDecodeError

        if response.status_code != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[InvoiceParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_, endpoint="/invoice/v1/processing")

        file_.close()
        standardize_response = eagledoc_invoice_parser(original_response)

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[ReceiptParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_, endpoint="/receipt/v3/processing")

        file_.close()

        standardize_response = eagledoc_receipt_parser(original_response)
        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )
