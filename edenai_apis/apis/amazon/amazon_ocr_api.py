import json
from io import BufferedReader
from typing import List, Sequence
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
from edenai_apis.utils.conversion import convert_string_to_number
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
        # Getting results from webhook.site
        data, *_ = check_webhook_result(job_id, self.api_settings)
        if data is None:
            return AsyncPendingResponseType[OcrTablesAsyncDataClass](
                provider_job_id=job_id
            )

        msg = json.loads(data.get("Message"))
        # ref: https://docs.aws.amazon.com/textract/latest/dg/async-notification-payload.html
        job_id = msg["JobId"]

        if msg["Status"] == "SUCCEEDED":
            original_result = self.clients["textract"].get_document_analysis(
                JobId=job_id
            )

            standardized_response = amazon_ocr_tables_parser(original_result)
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response=original_result,
                standardized_response=standardized_response,
                provider_job_id=job_id,
            )
        elif msg["Status"] == "PROCESSING":
            return AsyncPendingResponseType[OcrTablesAsyncDataClass](
                provider_job_id=job_id
            )

        else:
            original_result = self.clients["textract"].get_document_analysis(
                JobId=job_id
            )
            if original_result.get("JobStatus") == "FAILED":
                error = original_result.get("StatusMessage")
                raise ProviderException(error)

    def ocr__custom_document_parsing_async__launch_job(
        self, file: BufferedReader, queries: List[str]
    ) -> AsyncLaunchJobResponseType:

        file_content = file.read()

        self.storage_clients["textract"].Bucket(self.api_settings["bucket"]).put_object(
            Key=file.name, Body=file_content
        )

        formatted_queries = [{"Text": query, "Pages": ["1-*"]} for query in queries]

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

        response = self.clients["textract"].analyze_expense(
            Document={
                "S3Object": {"Bucket": self.api_settings["bucket"], "Name": file.name},
            }
        )

        extracted_data = []
        for invoice in response["ExpenseDocuments"]:

            # format response to be more easily parsable
            summary = {}
            currencies = {}
            for field in invoice["SummaryFields"]:
                field_type = field["Type"]["Text"]
                summary[field_type] = field["ValueDetection"]["Text"]
                field_currency = field.get("Currency", {}).get("Code")
                if field_currency is not None:
                    if field_currency not in currencies:
                        currencies[field_currency] = 1
                    else:
                        currencies[field_currency] += 1

            item_lines = []
            for line_item_group in invoice["LineItemGroups"]:
                for fields in line_item_group["LineItems"]:
                    parsed_items = {
                        item["Type"]["Text"]: item["ValueDetection"]["Text"]
                        for item in fields["LineItemExpenseFields"]
                    }
                    item_lines.append(
                        ItemLinesInvoice(
                            description=parsed_items.get("ITEM"),
                            quantity=parsed_items.get("QUANTITY"),
                            amount=convert_string_to_number(
                                parsed_items.get("PRICE"), float
                            ),
                            unit_price=convert_string_to_number(
                                parsed_items.get("UNIT_PRICE"), float
                            ),
                            discount=None,
                            product_code=parsed_items.get("PRODUCT_CODE"),
                            date_item=None,
                            tax_item=None,
                        )
                    )

            customer = CustomerInformationInvoice(
                customer_name=summary.get("RECEIVER_NAME", summary.get("NAME")),
                customer_address=summary.get(
                    "RECEIVER_ADDRESS", summary.get("ADDRESS")
                ),
                customer_email=None,
                customer_number=summary.get("CUSTOMER_NUMBER"),
                customer_tax_id=None,
                customer_mailing_address=None,
                customer_billing_address=None,
                customer_shipping_address=None,
                customer_service_address=None,
                customer_remittance_address=None,
            )

            merchant = MerchantInformationInvoice(
                merchant_name=summary.get("VENDOR_NAME"),
                merchant_address=summary.get("VENDOR_ADDRESS"),
                merchant_phone=summary.get("VENDOR_PHONE"),
                merchant_email=None,
                merchant_fax=None,
                merchant_website=summary.get("VENDOR_URL"),
                merchant_tax_id=summary.get("TAX_PAYER_ID"),
                merchant_siret=None,
                merchant_siren=None,
            )

            invoice_currency = None
            if len(currencies) == 1:
                invoice_currency = list(currencies.keys())[0]
            # HACK in case multiple currencies are returned,
            # we get the one who appeared the most
            elif len(currencies) > 1:
                invoice_currency = max(currencies, key=currencies.get)
            locale = LocaleInvoice(currency=invoice_currency, invoice_language=None)

            taxes = [
                TaxesInvoice(value=convert_string_to_number(summary.get("TAX"), float))
            ]

            invoice_infos = InfosInvoiceParserDataClass(
                customer_information=customer,
                merchant_information=merchant,
                invoice_number=summary.get("INVOICE_RECEIPT_ID"),
                invoice_total=convert_string_to_number(summary.get("TOTAL"), float),
                invoice_subtotal=convert_string_to_number(
                    summary.get("SUBTOTAL"), float
                ),
                amount_due=convert_string_to_number(summary.get("AMOUNT_DUE"), float),
                previous_unpaid_balance=summary.get("PRIOR_BALANCE"),
                discount=summary.get("DISCOUNT"),
                taxes=taxes,
                payment_term=summary.get("PAYMENT_TERMS"),
                purchase_order=None,
                date=summary.get("ORDER_DATE", summary.get("INVOICE_RECEIPT_DATE")),
                due_date=summary.get("DUE_DATE"),
                service_date=None,
                service_due_date=None,
                locale=locale,
                bank_information=None,
                item_lines=item_lines,
            )
            extracted_data.append(invoice_infos)

        return ResponseType(
            original_response=response,
            standardized_response=InvoiceParserDataClass(extracted_data=extracted_data),
        )
