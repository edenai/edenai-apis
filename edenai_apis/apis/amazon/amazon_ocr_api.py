import json
from time import sleep
from typing import List, Sequence, Dict, Union

from botocore.exceptions import ClientError

from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import (
    CustomDocumentParsingAsyncDataClass,
)
from edenai_apis.features.ocr.data_extraction.data_extraction_dataclass import (
    DataExtractionDataClass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    Country,
    IdentityParserDataClass,
    InfoCountry,
    InfosIdentityParserDataClass,
    ItemIdentityParserDataClass,
    format_date,
    get_info_country,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import Bounding_box, OcrDataClass
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import OcrAsyncDataClass
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.utils.async_to_sync import fibonacci_waiting_call
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
from .helpers import (
    amazon_data_extraction_formatter,
    amazon_ocr_async_formatter,
    amazon_ocr_tables_parser,
    amazon_custom_document_parsing_formatter,
    amazon_invoice_parser_formatter,
    amazon_receipt_parser_formatter,
    amazon_financial_parser_formatter,
    handle_amazon_call,
)


class AmazonOcrApi(OcrInterface):
    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        payload = {
            "Document": {
                "Bytes": file_content,
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file},
            }
        }
        response = handle_amazon_call(
            self.clients["textract"].detect_document_text, **payload
        )

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
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:

        with open(file, "rb") as file_:
            payload = {
                "DocumentPages": [
                    {
                        "Bytes": file_.read(),
                        "S3Object": {
                            "Bucket": self.api_settings["bucket"],
                            "Name": "test",
                        },
                    }
                ]
            }

            original_response = handle_amazon_call(
                self.clients["textract"].analyze_id, **payload
            )

        items: Sequence[InfosIdentityParserDataClass] = []
        for document in original_response["IdentityDocuments"]:
            infos: InfosIdentityParserDataClass = InfosIdentityParserDataClass.default()
            for field in document["IdentityDocumentFields"]:
                field_type = field["Type"]["Text"]
                confidence = round(field["ValueDetection"]["Confidence"] / 100, 2)
                value = (
                    field["ValueDetection"]["Text"]
                    if field["ValueDetection"]["Text"] != ""
                    else None
                )
                if field_type == "LAST_NAME":
                    infos.last_name = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type in ("FIRST_NAME", "MIDDLE_NAME") and value:
                    infos.given_names.append(
                        ItemIdentityParserDataClass(value=value, confidence=confidence)
                    )
                elif field_type == "DOCUMENT_NUMBER":
                    infos.document_id = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "EXPIRATION_DATE":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos.expire_date = ItemIdentityParserDataClass(
                        value=format_date(value),
                        confidence=confidence,
                    )
                elif field_type == "DATE_OF_BIRTH":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos.birth_date = ItemIdentityParserDataClass(
                        value=format_date(value),
                        confidence=confidence,
                    )
                elif field_type == "DATE_OF_ISSUE":
                    value = (
                        field["ValueDetection"].get("NormalizedValue", {}).get("Value")
                    )
                    infos.issuance_date = ItemIdentityParserDataClass(
                        value=format_date(value),
                        confidence=confidence,
                    )
                elif field_type == "ID_TYPE":
                    infos.document_type = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "ADDRESS":
                    infos.address = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )
                elif field_type == "COUNTY" and value:
                    infos.country = (
                        get_info_country(InfoCountry.NAME, value) or Country.default()
                    )
                    infos.country.confidence = confidence
                elif field_type == "MRZ_CODE":
                    infos.mrz = ItemIdentityParserDataClass(
                        value=value, confidence=confidence
                    )

            items.append(infos)

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: str, file_type: str, language: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        with open(file, "rb") as file_:
            file_content = file_.read()

        # upload file first
        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )

        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file},
            },
            "FeatureTypes": [
                "TABLES",
            ],
            "NotificationChannel": {
                "SNSTopicArn": self.api_settings["topic"],
                "RoleArn": self.api_settings["role"],
            },
        }

        response = handle_amazon_call(
            self.clients["textract"].start_document_analysis, **payload
        )

        return AsyncLaunchJobResponseType(provider_job_id=response["JobId"])

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:
        payload = {"JobId": job_id}
        response = handle_amazon_call(
            self.clients["textract"].get_document_analysis, **payload
        )

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
        self,
        file: str,
        queries: List[Dict[str, Union[str, str]]],
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        with open(file, "rb") as file_:
            file_content = file_.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )
        formatted_queries = [
            {
                "Text": query.get("query"),
                "Pages": str(query.get("pages", 1) or 1).split(",") or None,
            }
            for query in queries
        ]

        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file}
            },
            "FeatureTypes": ["QUERIES"],
            "QueriesConfig": {"Queries": formatted_queries},
        }
        response = handle_amazon_call(
            self.clients["textract"].start_document_analysis, **payload
        )

        return AsyncLaunchJobResponseType(provider_job_id=response["JobId"])

    def ocr__custom_document_parsing_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[CustomDocumentParsingAsyncDataClass]:
        try:
            response = self.clients["textract"].get_document_analysis(
                JobId=provider_job_id
            )
        except self.clients["image"].exceptions.InvalidParameterException as exc:
            raise ProviderException("Invalid Parameter: Only english are supported.")
        except ClientError as excp:
            response_meta = excp.response.get("ResponseMetadata", {}) or {}
            status_code = response_meta.get("HTTPStatusCode", None)
            if "Request has invalid Job Id" in str(excp):
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID, code=status_code
                )
            raise ProviderException(str(excp), code=status_code)

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
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )

        # Launch invoice job
        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file},
            }
        }
        launch_job_response = handle_amazon_call(
            self.clients["textract"].start_expense_analysis, **payload
        )

        # Get job result
        job_id = launch_job_response.get("JobId")
        get_response = self.clients["textract"].get_expense_analysis(JobId=job_id)

        if get_response["JobStatus"] == "FAILED":
            error: str = get_response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        waiting_args = {"JobId": job_id}
        get_response = fibonacci_waiting_call(
            max_time=60,
            status="SUCCEEDED",
            func=self.clients["textract"].get_expense_analysis,
            provider_handel_call=handle_amazon_call,
            **waiting_args,
        )  # waiting exponentially using fibonacci

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

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )

        # Launch invoice job
        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file}
            }
        }
        launch_job_response = handle_amazon_call(
            self.clients["textract"].start_expense_analysis, **payload
        )

        # Get job result
        job_id = launch_job_response.get("JobId")
        get_response = self.clients["textract"].get_expense_analysis(JobId=job_id)

        if get_response["JobStatus"] == "FAILED":
            error: str = get_response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        waiting_args = {"JobId": job_id}
        get_response = fibonacci_waiting_call(
            max_time=60,
            status="SUCCEEDED",
            func=self.clients["textract"].get_expense_analysis,
            provider_handel_call=handle_amazon_call,
            **waiting_args,
        )

        # Check if NextToken exist
        pagination_token = get_response.get("NextToken")
        pages = [get_response]
        if not pagination_token:
            return ResponseType(
                original_response=pages,
                standardized_response=amazon_receipt_parser_formatter(pages),
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
            standardized_response=amazon_receipt_parser_formatter(pages),
        )

    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        with open(file, "rb") as file_:
            file_content = file_.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )

        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file}
            }
        }
        launch_job_response = handle_amazon_call(
            self.clients["textract"].start_document_text_detection, **payload
        )

        return AsyncLaunchJobResponseType(provider_job_id=launch_job_response["JobId"])

    def ocr__ocr_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrAsyncDataClass]:
        payload = {"JobId": provider_job_id}
        response = handle_amazon_call(
            self.clients["textract"].get_document_text_detection, **payload
        )

        if response["JobStatus"] == "FAILED":
            error: str = response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        if response["JobStatus"] == "SUCCEEDED":
            pagination_token = response.get("NextToken")
            responses = [response]

            while pagination_token:
                payload = {
                    "JobId": provider_job_id,
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["textract"].get_document_text_detection, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)
                pagination_token = response.get("NextToken")

            return AsyncResponseType(
                original_response=responses,
                standardized_response=amazon_ocr_async_formatter(responses),
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType(provider_job_id=response["JobStatus"])

    def ocr__data_extraction(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[DataExtractionDataClass]:
        with open(file, "rb") as fstream:
            file_content = fstream.read()

            self.storage_clients["textract"].Bucket(
                self.api_settings["bucket"]
            ).put_object(Key=file, Body=file_content)

            payload = {
                "DocumentLocation": {
                    "S3Object": {
                        "Bucket": self.api_settings["bucket"],
                        "Name": file,
                    }
                },
                "FeatureTypes": ["FORMS"],
            }
            launch_job_response = handle_amazon_call(
                self.clients["textract"].start_document_analysis, **payload
            )

            waiting_args = {"JobId": launch_job_response["JobId"]}
            response = fibonacci_waiting_call(
                max_time=100,
                status="IN_PROGRESS",
                func=self.clients["textract"].get_document_analysis,
                provider_handel_call=handle_amazon_call,
                status_positif=False,
                **waiting_args,
            )

            if response["JobStatus"] == "FAILED":
                error: str = response.get(
                    "StatusMessage", "Amazon returned a job status: FAILED"
                )
                raise ProviderException(error)

            responses = [response]

            standardized_response = amazon_data_extraction_formatter(responses)

            while pagination_token := response.get("NextToken"):
                payload = {
                    "JobId": launch_job_response["JobId"],
                    "NextToken": pagination_token,
                }
                response = handle_amazon_call(
                    self.clients["textract"].get_document_analysis, **payload
                )

                if response["JobStatus"] == "FAILED":
                    error: str = response.get(
                        "StatusMessage", "Amazon returned a job status: FAILED"
                    )
                    raise ProviderException(error)

                responses.append(response)

            return ResponseType[DataExtractionDataClass](
                original_response=response,
                standardized_response=standardized_response,
            )

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str,
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file, Body=file_content
        )

        payload = {
            "DocumentLocation": {
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file},
            }
        }
        launch_job_response = handle_amazon_call(
            self.clients["textract"].start_expense_analysis, **payload
        )

        # Get job result
        job_id = launch_job_response.get("JobId")
        get_response = self.clients["textract"].get_expense_analysis(JobId=job_id)

        if get_response["JobStatus"] == "FAILED":
            error: str = get_response.get(
                "StatusMessage", "Amazon returned a job status: FAILED"
            )
            raise ProviderException(error)

        waiting_args = {"JobId": job_id}
        get_response = fibonacci_waiting_call(
            max_time=60,
            status="SUCCEEDED",
            func=self.clients["textract"].get_expense_analysis,
            provider_handel_call=handle_amazon_call,
            **waiting_args,
        )  # waiting exponentially using fibonacci

        # Check if NextToken exist
        pagination_token = get_response.get("NextToken")
        pages = [get_response]
        if not pagination_token:
            return ResponseType(
                original_response=pages,
                standardized_response=amazon_financial_parser_formatter(pages),
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
            standardized_response=amazon_financial_parser_formatter(pages),
        )
