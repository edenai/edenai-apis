from typing import Dict

from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import (
    ResumeParserDataClass,
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
    FinancialParserType,
)
from edenai_apis.features.ocr.identity_parser import IdentityParserDataClass
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from .client import Client
from .document import FileParameter, UploadDocumentParams
from .standardization import (
    IdentityStandardizer,
    InvoiceStandardizer,
    ReceiptStandardizer,
    ResumeStandardizer,
    FinancialStandardizer,
)


class AffindaApi(ProviderInterface, OcrInterface):
    provider_name = "affinda"

    def __init__(self, api_keys: Dict = {}):
        super().__init__()
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        self.client = Client(self.api_settings["api_key"])
        self.client.current_organization = self.client.get_organizations()[0].identifier

    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:
        self.client.current_workspace = self.api_settings["nextgen_resume_parser"]

        document = self.client.create_document(
            file=FileParameter(file=file, url=file_url)
        )
        original_response = self.client.last_api_response

        standardizer = ResumeStandardizer(document=document)
        standardizer.std_personnal_information()
        standardizer.std_education()
        standardizer.std_work_experience()
        standardizer.std_skills()
        standardizer.std_miscellaneous()

        self.client.delete_document(document.meta.identifier)

        return ResponseType[ResumeParserDataClass](
            original_response=original_response,
            standardized_response=standardizer.standardized_response,
        )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        self.client.current_workspace = self.api_settings["invoice_workspace"]

        document = self.client.create_document(
            file=FileParameter(file=file, url=file_url)
        )
        original_response = self.client.last_api_response

        standardizer = InvoiceStandardizer(document=document)
        standardizer.std_merchant_informations()
        standardizer.std_customer_information()
        standardizer.std_invoice_informations()
        standardizer.std_dates_informations()
        standardizer.std_bank_information()
        standardizer.std_taxes_informations()
        standardizer.std_items_lines_informations()

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardizer.standardized_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        self.client.current_workspace = self.api_settings["receipt_workspace"]
        document = self.client.create_document(
            file=FileParameter(file=file, url=file_url),
            parameters=UploadDocumentParams(language=language),
        )
        original_response = self.client.last_api_response

        standardizer = ReceiptStandardizer(document=document)
        standardizer.std_merchant_informations()
        standardizer.std_payment_informations()
        standardizer.std_locale_information()
        standardizer.std__taxes_informations()
        standardizer.std_miscellaneous()
        standardizer.std_item_lines()

        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardizer.standardized_response,
        )

    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:
        self.client.current_workspace = self.api_settings["identity_workspace"]
        document = self.client.create_document(
            file=FileParameter(file=file, url=file_url)
        )
        original_response = self.client.last_api_response

        standardizer = IdentityStandardizer(document=document)
        standardizer.std_names_information()
        standardizer.std_document_information()
        standardizer.std_location_information()

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardizer.standardized_response,
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
        workspace_key = (
            "receipt_workspace"
            if document_type == FinancialParserType.RECEIPT.value
            else "invoice_workspace"
        )
        self.client.current_workspace = self.api_settings[workspace_key]
        document = self.client.create_document(
            file=FileParameter(file=file, url=file_url)
        )
        original_response = self.client.last_api_response
        standardizer = FinancialStandardizer(
            document=document, original_response=original_response
        )
        standardizer.std_response()

        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardizer.standardized_response,
        )
