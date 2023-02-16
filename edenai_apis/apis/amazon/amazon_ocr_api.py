import json
from io import BufferedReader
from pprint import pprint
from time import sleep
from typing import List, Sequence, Dict, Union
from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import (
    CustomDocumentParsingAsyncDataClass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    IdentityParserDataClass,
    InfoCountry,
    ItemIdentityParserDataClass,
    format_date,
    get_info_country,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import Bounding_box, OcrDataClass
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)

from .helpers import (
    check_webhook_result,
    amazon_ocr_tables_parser,
    amazon_custom_document_parsing_formatter,
    amazon_invoice_parser_formatter,
)


class AmazonOcrApi(OcrInterface):
    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        file_content = file.read()

        try:
            response = self.clients["textract"].detect_document_text(
                Document={
                    "Bytes": file_content,
                    "S3Object": {
                        "Bucket": self.api_settings["bucket"],
                        "Name": file.name,
                    },
                }
            )
        except Exception as amazon_call_exception:
            raise ProviderException(str(amazon_call_exception))

        final_text = ""
        output_value = json.dumps(response, ensure_ascii=False)
        original_response = json.loads(output_value)
        boxes: Sequence[Bounding_box] = []

        # Get region of text
        for region in original_response.get("Blocks"):
            if region.get("BlockType") == "LINE":
                # Read line by region
                final_text += " " + region.get("Text")

            if region.get("BlockType") == "WORD":
                boxes.append(
                    Bounding_box(
                        text=region.get("Text"),
                        left=region["Geometry"]["BoundingBox"]["Left"],
                        top=region["Geometry"]["BoundingBox"]["Top"],
                        width=region["Geometry"]["BoundingBox"]["Width"],
                        height=region["Geometry"]["BoundingBox"]["Height"],
                    )
                )

        standardized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )

        return ResponseType[OcrDataClass](
            original_response=original_response, standardized_response=standardized
        )

    def ocr__identity_parser(
        self, file: BufferedReader
    ) -> ResponseType[IdentityParserDataClass]:
        original_response = self.clients["textract"].analyze_id(
            DocumentPages=[
                {
                    "Bytes": file.read(),
                    "S3Object": {"Bucket": self.api_settings["bucket"], "Name": "test"},
                }
            ]
        )

        items = []
        for document in original_response["IdentityDocuments"]:
            infos = {}
            infos["given_names"] = []
            for field in document["IdentityDocumentFields"]:
                field_type = field["Type"]["Text"]
                confidence = round(field["ValueDetection"]["Confidence"] / 100, 2)
                value = (
                    field["ValueDetection"]["Text"]
                    if field["ValueDetection"]["Text"] != ""
                    else None
                )
                if field_type == "LAST_NAME":
                    infos["last_name"] = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type in ("FIRST_NAME", "MIDDLE_NAME") and value:
                    infos["given_names"].append(
                        ItemIdentityParserDataClass(value=value, confidence=confidence)
                    )
                elif field_type == "DOCUMENT_NUMBER":
                    infos["document_id"] = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "EXPIRATION_DATE":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos["expire_date"] = ItemIdentityParserDataClass(
                        value=format_date(value, "%Y-%m-%dT%H:%M:%S"),
                        confidence=confidence,
                    )
                elif field_type == "DATE_OF_BIRTH":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos["birth_date"] = ItemIdentityParserDataClass(
                        value=format_date(value, "%Y-%m-%dT%H:%M:%S"),
                        confidence=confidence,
                    )
                elif field_type == "DATE_OF_ISSUE":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos["issuance_date"] = ItemIdentityParserDataClass(
                        value=format_date(value, "%Y-%m-%dT%H:%M:%S"),
                        confidence=confidence,
                    )
                elif field_type == "ID_TYPE":
                    infos["document_type"] = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "ADDRESS":
                    infos["address"] = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "COUNTY" and value:
                    infos["country"] = get_info_country(InfoCountry.NAME, value)
                    infos["country"]["confidence"] = confidence
                elif field_type == "MRZ_CODE":
                    infos["mrz"] = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )

            items.append(infos)

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: BufferedReader, file_type: str, language: str
    ) -> AsyncLaunchJobResponseType:
        file_content = file.read()

        # upload file first
        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file.name, Body=file_content
        )

        response = self.clients["textract"].start_document_analysis(
            DocumentLocation={
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file.name},
            },
            FeatureTypes=[
                "TABLES",
            ],
            NotificationChannel={
                "SNSTopicArn": self.api_settings["topic"],
                "RoleArn": self.api_settings["role"],
            },
        )

        return AsyncLaunchJobResponseType(provider_job_id=response["JobId"])

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:
        response = self.clients["textract"].get_document_analysis(JobId=job_id)

        if response.get("JobStatus") == "IN_PROGRESS":
            return AsyncPendingResponseType[OcrTablesAsyncDataClass](
                provider_job_id=job_id
            )
        elif response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        pagination_token = response.get("NextToken")
        pages = [response]
        if not pagination_token:
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response=pages,
                standardized_response=amazon_ocr_tables_parser(pages),
                provider_job_id=job_id,
            )

        finished = False
        while not finished:
            response = self.clients["textract"].get_document_analysis(
                JobId=job_id,
                NextToken=pagination_token,
            )
            pages.append(response)
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return AsyncResponseType[OcrTablesAsyncDataClass](
            original_response=pages,
            standardized_response=amazon_ocr_tables_parser(pages),
            provider_job_id=job_id,
        )

    def ocr__custom_document_parsing_async__launch_job(
        self, file: BufferedReader, queries: List[Dict[str, Union[str, str]]],
    ) -> AsyncLaunchJobResponseType:

        file_content = file.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file.name, Body=file_content
        )
        formatted_queries = [{"Text": query.get("query"), "Pages" : query.get("pages").split(',')} for query in queries]

        try:
            response = self.clients["textract"].start_document_analysis(
                DocumentLocation={
                    "S3Object": {
                        "Bucket": self.api_settings["bucket"],
                        "Name": file.name,
                    },
                },
                FeatureTypes=["QUERIES"],
                QueriesConfig={"Queries": formatted_queries},
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        return AsyncLaunchJobResponseType(provider_job_id=response["JobId"])

    def ocr__custom_document_parsing_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[CustomDocumentParsingAsyncDataClass]:
        response = self.clients["textract"].get_document_analysis(JobId=provider_job_id)

        if response.get("JobStatus") == "IN_PROGRESS":
            return AsyncPendingResponseType[CustomDocumentParsingAsyncDataClass](
                provider_job_id=provider_job_id
            )
        elif response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        pagination_token = response.get("NextToken")
        pages = [response]
        if not pagination_token:
            return AsyncResponseType[CustomDocumentParsingAsyncDataClass](
                original_response=pages,
                standardized_response=amazon_custom_document_parsing_formatter(pages),
                provider_job_id=provider_job_id,
            )

        finished = False
        while not finished:
            response = self.clients["textract"].get_document_analysis(
                JobId=provider_job_id,
                NextToken=pagination_token,
            )
            pages.append(response)
            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                finished = True

        return AsyncResponseType[CustomDocumentParsingAsyncDataClass](
            original_response=pages,
            standardized_response=amazon_custom_document_parsing_formatter(pages),
            provider_job_id=provider_job_id,
        )

    def ocr__invoice_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[InvoiceParserDataClass]:
        file_content = file.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file.name, Body=file_content
        )

        # Launch invoice job 
        try:
            launch_job_response = self.clients["textract"].start_expense_analysis(
                DocumentLocation={
                    "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file.name},
                }
            )
        except Exception as amazon_call_exception:
            raise ProviderException(str(amazon_call_exception))
        
        # Get job result
        job_id = launch_job_response.get('JobId')
        get_response = self.clients["textract"].get_expense_analysis(JobId=job_id)
        
        if get_response["JobStatus"] == "FAILED":
            error: str = get_response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)
        
        wait_time = 0
        while wait_time < 60:  # Wait for the answer from provider
            if get_response['JobStatus'] == "SUCCEEDED":
                break
            sleep(3)
            wait_time += 3
            get_response = self.clients["textract"].get_expense_analysis(JobId=job_id)
        
        # Check if NextToken exist
        pagination_token = get_response.get("NextToken")
        pages = [get_response]
        if not pagination_token:
            return ResponseType(
                original_response=pages,
                standardized_response=amazon_invoice_parser_formatter(pages),
            )

        finished = False
        while not finished:
            get_response = self.clients["textract"].get_expense_analysis(
                JobId=job_id,
                NextToken=pagination_token,
            )
            pages.append(get_response)
            if "NextToken" in get_response:
                pagination_token = get_response["NextToken"]
            else:
                finished = True
        
        return ResponseType(
            original_response=pages,
            standardized_response=amazon_invoice_parser_formatter(pages),
        )
