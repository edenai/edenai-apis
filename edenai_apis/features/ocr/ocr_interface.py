from abc import abstractmethod
from io import BufferedReader
from typing import List
from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import (
    CustomDocumentParsingAsyncDataclass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    IdentityParserDataClass,
)

from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import OcrDataClass
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
)
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import (
    ResumeParserDataClass,
)
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType, ResponseType


class OcrInterface:
    @abstractmethod
    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        """Optical Character Recognition on a file

        Args:
            file (BufferedReader): document to analyze
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__invoice_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[InvoiceParserDataClass]:
        """Parse an invoice and returned structured data

        Args:
            file (BufferedReader): invoice to analyze
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__ocr_tables_async__launch_job(
        self, file: BufferedReader, file_type: str, language: str
    ) -> AsyncLaunchJobResponseType:
        """Launch an asynchronous job to analyze tables in document
        Args:
            file (BufferedReader): document file to analyze
            file_type (file_type): document file to analyze
            language (str): language code in ISO format
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
        Get the result of an asynchrous job from webhook

        Args:
            - data (dict): result data given by the provider
            when calling the webhook
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__receipt_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[ReceiptParserDataClass]:
        """Parse a receipt and returned structured data

        Args:
            file (BufferedReader): receipt to analyze
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__resume_parser(
        self, file: BufferedReader
    ) -> ResponseType[ResumeParserDataClass]:
        """Parse a resume and returned structured data

        Args:
            file (BufferedReader): resume to analyze
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__identity_parser(
        self,
        file: BufferedReader,
    ) -> ResponseType[IdentityParserDataClass]:
        """Parse a identity document and returned structured data

        Args:
            file (BufferedReader): resume to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__custom_document_parsing_async__launch_job(
        self, file: BufferedReader, queries: List[str]
    ) -> AsyncLaunchJobResponseType:
        """
        Parse a document and extract data according to queries

        Args:
            file (BufferedReader): document to analyze
            queries (list[str]): list of queries describing what to extract from the document.
        """
        raise NotImplementedError

    @abstractmethod
    def ocr__custom_document_parsing_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[CustomDocumentParsingAsyncDataclass]:
        """
        Get the result of an asynchronous job by its ID

        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError
