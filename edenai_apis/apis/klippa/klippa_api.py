from io import BufferedReader
from json import JSONDecodeError
from typing import Dict

import requests

from edenai_apis.features import OcrInterface, ProviderInterface
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import FinancialParserDataClass
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import IdentityParserDataClass
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.features.ocr.resume_parser import ResumeParserDataClass
from edenai_apis.loaders.loaders import ProviderDataEnum, load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.klippa.klippa_ocr_normalizer import (
    klippa_invoice_parser,
    klippa_financial_parser,
    klippa_id_parser,
    klippa_receipt_parser,
    klippa_resume_parser
)

class KlippaApi(ProviderInterface, OcrInterface):
    provider_name = "klippa"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["subscription_key"]
        self.url = "https://custom-ocr.klippa.com/api/v1/parseDocument"

        self.headers = {
            "X-Auth-Key": self.api_key,
        }

    def _make_post_request(self, file: BufferedReader, endpoint: str = ""):
        files = {
            "document": file,
        }
        data = {"pdf_text_extraction": "full"}
        response = requests.post(
            url=self.url + endpoint, headers=self.headers, files=files, data=data
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
        original_response = self._make_post_request(file_)

        file_.close()
        standardize_response = klippa_invoice_parser(original_response)

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[ReceiptParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)

        file_.close()

        standardize_response = klippa_receipt_parser(original_response)
        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response,
        )

    def ocr__identity_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[IdentityParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_, endpoint="/identity")
        file_.close()

        standardized_response = klippa_id_parser(original_response)
        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__resume_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[ResumeParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_, endpoint="/resume")
        file_.close()
        
        standardized_response = klippa_resume_parser(original_response)
        return ResponseType[ResumeParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
    
    def ocr__financial_parser(
        self, file: str, language: str, document_type: str = "", file_url: str = ""
    ) -> ResponseType[FinancialParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)

        file_.close()

        standardize_response = klippa_financial_parser(original_response)

        return ResponseType[FinancialParserDataClass](
                    original_response=original_response,
                    standardized_response=standardize_response,
                )
