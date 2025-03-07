from abc import ABC, abstractmethod
from typing import List, Dict, Union

from edenai_apis.features.ocr.anonymization_async.anonymization_async_dataclass import (
    AnonymizationAsyncDataClass,
)
from edenai_apis.features.ocr.bank_check_parsing.bank_check_parsing_dataclass import (
    BankCheckParsingDataClass,
)
from edenai_apis.features.ocr.custom_document_parsing_async import (
    CustomDocumentParsingAsyncDataClass,
)
from edenai_apis.features.ocr.data_extraction.data_extraction_dataclass import (
    DataExtractionDataClass,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    IdentityParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.invoice_splitter_async.invoice_splitter_async_dataclass import (
    InvoiceSplitterAsyncDataClass,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import OcrDataClass
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import OcrAsyncDataClass
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
)
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import (
    ResumeParserDataClass,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    ResponseType,
)


class OcrInterface:
    @abstractmethod
    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        """Optical Character Recognition on a file

        Args:
            file (BufferedReader): document to analyze
            file_url (str, optional): url of file
            language (str): language code in ISO format
        """
        raise NotImplementedError

    # DEPRECATED
    @abstractmethod
    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        """Parse an invoice and returned structured data

        Args:
            file (BufferedReader): invoice to analyze
            file_url (str, optional): url of file
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_tables_async__launch_job(
        self, file: str, file_type: str, language: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """Launch an asynchronous job to analyze tables in document
        Args:
            file (BufferedReader): document file to analyze
            file_type (file_type): document file to analyze
            language (str): language code in ISO format
            file_url (str, optional): url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> ResponseType[OcrTablesAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_tables_async__get_results_from_webhook(
        self, data: dict
    ) -> ResponseType[OcrTablesAsyncDataClass]:
        """
        Get the result of an asynchronous job from webhook

        Args:
            - data (dict): result data given by the provider
            when calling the webhook
        """
        raise NotImplementedError

    # DEPRECATED
    @abstractmethod
    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        """Parse a receipt and returned structured data

        Args:
            file (BufferedReader): receipt to analyze
            file_url (str, optional): url of file
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        """Parse a financial document (receipt or invoice) and returned structured data

        Args:
            file (BufferedReader): receipt to analyze
            file_url (str, optional): url of file
            language (str): language code in ISO format
            document_type (str): type of the document (invoice or receipt) can be auto-detect for some providers
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:
        """Parse a resume and returned structured data

        Args:
            file (BufferedReader): resume to analyze
            file_url (str, optional): url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:
        """Parse an identity document and returned structured data

        Args:
            file (BufferedReader): resume to analyze
            file_url (str, optional): url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__custom_document_parsing_async__launch_job(
        self,
        file: str,
        queries: List[Dict[str, Union[str, str]]],
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        """
        Parse a document and extract data according to queries

        Args:
            file (str): document to analyze
            file_url (str): url of file
            queries (list[str]): list of queries describing what to extract from the document.
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__custom_document_parsing_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[CustomDocumentParsingAsyncDataClass]:
        """
        Get the result of an asynchronous job by its ID

        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """Optical Character Recognition on a file

        Args:
            file (BufferedReader): document to analyze
            file_url (str, optional): url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrAsyncDataClass]:
        """
        Get the result of an asynchronous job by its ID

        Args:
            provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__data_extraction(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[DataExtractionDataClass]:
        """
        Parse a document and extract all information
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__bank_check_parsing(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[BankCheckParsingDataClass]:
        """
        Parse a bank check and extract all information

        Args:
            file (str): Path of file
            file_url (str, optional): Url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__anonymization_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Parse a document and returns anonymized document

        Args:
            file (str): Path of file
            file_url (str) : Url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__anonymization_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AnonymizationAsyncDataClass]:
        """
        Get the result of an asynchronous job by its ID

        Args:
            provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__invoice_splitter_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        """
        Split an invoice into multiple invoices

        Args:
            file (str): Path of file
            file_url (str) : Url of file
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__invoice_splitter_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[InvoiceSplitterAsyncDataClass]:
        """
        Get the result of an asynchronous job by its ID

        Args:
            provider_job_id (str): id of async job
        """
        raise NotImplementedError
