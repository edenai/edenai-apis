import json
from collections import defaultdict
from typing import Sequence

import requests
from PIL import Image as Img
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from edenai_apis.apis.microsoft.microsoft_helpers import (
    microsoft_ocr_tables_standardize_response,
    normalize_invoice_result,
    get_microsoft_urls,
    microsoft_financial_parser_formatter,
    microsoft_ocr_async_standardize_response,
)
from edenai_apis.features.ocr import (
    Bounding_box,
    IdentityParserDataClass,
    InfosIdentityParserDataClass,
    InfosReceiptParserDataClass,
    InvoiceParserDataClass,
    ItemLines,
    # MerchantInformation,
    OcrDataClass,
    PaymentInformation,
    ReceiptParserDataClass,
    Taxes,
    get_info_country,
    OcrAsyncDataClass,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
    FinancialParserType,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    Country,
    InfoCountry,
    ItemIdentityParserDataClass,
    format_date,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    MerchantInformation,
)
from edenai_apis.utils.conversion import add_query_param_in_url
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


class MicrosoftOcrApi(OcrInterface):
    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        url = f"{self.api_settings['vision']['url']}/ocr?detectOrientation=true"

        request = requests.post(
            url=add_query_param_in_url(url, {"language": language}),
            headers=self.headers["vision"],
            data=file_content,
        )
        response = request.json()

        final_text = ""

        if "error" in response:
            raise ProviderException(response["error"]["message"], request.status_code)

        # Get width and hight

        with Img.open(file) as img:
            width, height = img.size
        boxes: Sequence[Bounding_box] = []
        # Get region of text
        for region in response["regions"]:
            # Read line by region
            for line in region["lines"]:
                for word in line["words"]:
                    final_text += " " + word["text"]
                    boxes.append(
                        Bounding_box(
                            text=word["text"],
                            left=float(word["boundingBox"].split(",")[0]) / width,
                            top=float(word["boundingBox"].split(",")[1]) / height,
                            width=float(word["boundingBox"].split(",")[2]) / width,
                            height=float(word["boundingBox"].split(",")[3]) / height,
                        )
                    )
        standardized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )

        return ResponseType[OcrDataClass](
            original_response=response, standardized_response=standardized
        )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        with open(file, "rb") as file_:
            try:
                document_analysis_client = DocumentAnalysisClient(
                    endpoint=self.url["documentintelligence"],
                    credential=AzureKeyCredential(
                        self.api_settings["documentintelligence"]["subscription_key"]
                    ),
                )
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-invoice", file_
                )
                invoices = poller.result()
            except AzureError as provider_call_exception:
                raise ProviderException(str(provider_call_exception))

            try:
                if invoices is None or not hasattr(invoices, "to_dict"):
                    raise AttributeError
                # AttributeError sometimes happens in the lib when calling to dict
                # and a DocumentField has a None value
                original_response = invoices.to_dict()
            except AttributeError:
                raise ProviderException("Provider return an empty response")

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=normalize_invoice_result(original_response),
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        with open(file, "rb") as file_:
            try:
                document_analysis_client = DocumentAnalysisClient(
                    endpoint=self.url["documentintelligence"],
                    credential=AzureKeyCredential(
                        self.api_settings["documentintelligence"]["subscription_key"]
                    ),
                )
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-receipt", file_
                )
                form_pages = poller.result()
            except AzureError as provider_call_exception:
                raise ProviderException(str(provider_call_exception))

            if form_pages is None or not hasattr(form_pages, "to_dict"):
                raise ProviderException("Provider return an empty response")
            original_response = form_pages.to_dict()

        # Normalize the response
        default_dict = defaultdict(lambda: None)
        receipts = []
        for fields in original_response.get("documents", []):
            fields = fields.get("fields")
            if not fields:
                continue

            # 1. Receipt Total
            receipt_total = fields.get("Total", default_dict).get("value")

            # 2. Date & time
            date = fields.get("TransactionDate", default_dict).get("value")
            time = fields.get("TransactionTime", default_dict).get("value")

            # 3. receipt_subtotal
            sub_total = fields.get("Subtotal", default_dict).get("value")

            # 4. merchant informations
            merchant = MerchantInformation(
                merchant_name=fields.get("MerchantName", default_dict).get("value"),
                merchant_address=fields.get("MerchantAddress", default_dict).get(
                    "content"
                ),
                merchant_phone=fields.get("MerchantPhoneNumber", default_dict).get(
                    "value"
                ),
                merchant_url=None,
                merchant_siren=None,
                merchant_siret=None,
            )

            # 5. Taxes
            taxes = [Taxes(taxes=fields.get("Tax", default_dict).get("value"))]

            # 6. Receipt infos / payment informations
            receipt_infos = {"doc_type": fields.get("doc_type")}
            try:
                tip_float = fields.get("Tip", default_dict).get("value")
                tip = str(tip_float) if tip_float else None
            except:
                tip = None
            payment_infos = PaymentInformation(tip=tip)

            # 7. Items
            items = []
            for item in fields.get("Items", default_dict).get("value", []):
                description = item["value"].get("Name", default_dict).get("value")
                price = item["value"].get("Price", default_dict).get("value")
                quantity_str = item["value"].get("Quantity", default_dict).get("value")
                quantity = float(quantity_str) if quantity_str else None
                total = item["value"].get("TotalPrice", default_dict).get("value")
                items.append(
                    ItemLines(
                        amount=total,
                        description=description,
                        unit_price=price,
                        quantity=quantity,
                    )
                )

            receipts.append(
                InfosReceiptParserDataClass(
                    item_lines=items,
                    taxes=taxes,
                    merchant_information=merchant,
                    invoice_subtotal=sub_total,
                    receipt_total=receipt_total,
                    date=str(date),
                    time=str(time),
                    payment_information=payment_infos,
                    receipt_infos=receipt_infos,
                )
            )
        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=ReceiptParserDataClass(extracted_data=receipts),
        )

    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:
        with open(file, "rb") as file_:
            try:
                document_analysis_client = DocumentAnalysisClient(
                    endpoint=self.url["documentintelligence"],
                    credential=AzureKeyCredential(
                        self.api_settings["documentintelligence"]["subscription_key"]
                    ),
                )
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-idDocument", file_
                )
                response = poller.result()
            except AzureError as provider_call_exception:
                raise ProviderException(str(provider_call_exception))

            if response is None or not hasattr(response, "to_dict"):
                raise ProviderException("Provider return an empty response")
            original_response = response.to_dict()

        items = []

        for document in original_response.get("documents", []):
            fields = document["fields"]
            country = get_info_country(
                key=InfoCountry.ALPHA3,
                value=fields.get("CountryRegion", {}).get("content"),
            )
            if country:
                country["confidence"] = fields.get("CountryRegion", {}).get(
                    "confidence"
                )

            given_names = fields.get("FirstName", {}).get("content", "").split(" ")
            final_given_names = []
            for given_name in given_names:
                final_given_names.append(
                    ItemIdentityParserDataClass(
                        value=given_name,
                        confidence=fields.get("FirstName", {}).get("confidence"),
                    )
                )

            items.append(
                InfosIdentityParserDataClass(
                    document_type=ItemIdentityParserDataClass(
                        value=document.get("docType"),
                        confidence=document.get("confidence"),
                    ),
                    country=country or Country.default(),
                    birth_date=ItemIdentityParserDataClass(
                        value=format_date(fields.get("DateOfBirth", {}).get("value")),
                        confidence=fields.get("DateOfBirth", {}).get("confidence"),
                    ),
                    expire_date=ItemIdentityParserDataClass(
                        value=format_date(
                            fields.get("DateOfExpiration", {}).get("value")
                        ),
                        confidence=fields.get("DateOfExpiration", {}).get("confidence"),
                    ),
                    issuance_date=ItemIdentityParserDataClass(
                        value=format_date(fields.get("DateOfIssue", {}).get("value")),
                        confidence=fields.get("DateOfIssue", {}).get("confidence"),
                    ),
                    issuing_state=ItemIdentityParserDataClass(
                        value=fields.get("IssuingAuthority", {}).get("content"),
                        confidence=fields.get("IssuingAuthority", {}).get("confidence"),
                    ),
                    document_id=ItemIdentityParserDataClass(
                        value=fields.get("DocumentNumber", {}).get("content"),
                        confidence=fields.get("DocumentNumber", {}).get("confidence"),
                    ),
                    last_name=ItemIdentityParserDataClass(
                        value=fields.get("LastName", {}).get("content"),
                        confidence=fields.get("LastName", {}).get("confidence"),
                    ),
                    given_names=final_given_names,
                    mrz=ItemIdentityParserDataClass(
                        value=fields.get("MachineReadableZone", {}).get("content"),
                        confidence=fields.get("MachineReadableZone", {}).get(
                            "confidence"
                        ),
                    ),
                    nationality=ItemIdentityParserDataClass(
                        value=fields.get("Nationality", {}).get("content"),
                        confidence=fields.get("Nationality", {}).get("confidence"),
                    ),
                    birth_place=ItemIdentityParserDataClass(
                        value=fields.get("PlaceOfBirth", {}).get("content"),
                        confidence=fields.get("PlaceOfBirth", {}).get("confidence"),
                    ),
                    gender=ItemIdentityParserDataClass(
                        value=fields.get("Sex", {}).get("content"),
                        confidence=fields.get("Sex", {}).get("confidence"),
                    ),
                    address=ItemIdentityParserDataClass(),
                    age=ItemIdentityParserDataClass(),
                    image_id=[],
                    image_signature=[],
                )
            )

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
        url = (
            f"{self.url['documentintelligence']}documentintelligence/documentModels/"
            f"prebuilt-layout:analyze?api-version=2024-11-30"
        )
        url = add_query_param_in_url(url, {"locale": language})

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": self.api_settings["documentintelligence"][
                    "subscription_key"
                ],
            },
            data=file_content,
        )

        if response.status_code != 202:
            error = response.json()["error"]["innererror"]["message"]
            raise ProviderException(error, code=response.status_code)

        return AsyncLaunchJobResponseType(
            provider_job_id=response.headers.get("apim-request-id")
        )

    def ocr__ocr_tables_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:
        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.api_settings["documentintelligence"][
                "subscription_key"
            ],
        }

        url = (
            self.url["documentintelligence"]
            + f"documentintelligence/documentModels/prebuilt-layout/"
            f"analyzeResults/{provider_job_id}?api-version=2024-11-30"
        )
        response = requests.get(url, headers=headers)
        if response.status_code >= 400:
            try:
                error = response.json()["error"]["message"]
                if "Resource not found" in error:
                    raise AsyncJobException(
                        reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                        code=response.status_code,
                    )
                raise ProviderException(error, code=response.status_code)
            except (KeyError, json.JSONDecodeError) as exc:
                raise ProviderException(
                    message=response.text, code=response.status_code
                ) from exc

        data = response.json()
        if data.get("error"):
            raise ProviderException(data.get("error"), code=response.status_code)
        if data["status"] == "succeeded":
            original_result = data["analyzeResult"]
            standardized_response = microsoft_ocr_tables_standardize_response(
                original_result
            )
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response=data,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[OcrTablesAsyncDataClass](
            provider_job_id=provider_job_id
        )

    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        with open(file, "rb") as file_:
            file_content = file_.read()

        url = (
            f"{self.url['documentintelligence']}documentintelligence/documentModels/"
            f"prebuilt-layout:analyze?api-version=2024-02-29-preview"
        )
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": self.api_settings["documentintelligence"][
                    "subscription_key"
                ],
            },
            data=file_content,
        )
        if response.status_code != 202:
            error = response.json()["error"]["innererror"]["message"]
            raise ProviderException(error, code=response.status_code)
        return AsyncLaunchJobResponseType(
            provider_job_id=response.headers.get("apim-request-id")
        )

    def ocr__ocr_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrDataClass]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_settings["documentintelligence"][
                "subscription_key"
            ],
        }

        url = (
            self.url["documentintelligence"]
            + f"documentintelligence/documentModels/prebuilt-layout/"
            f"analyzeResults/{provider_job_id}?api-version=2024-02-29-preview"
        )
        response = requests.get(url, headers=headers)

        if response.status_code >= 400:
            error = response.json()["error"]["message"]
            if "Resource not found" in error:
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                    code=response.status_code,
                )
            raise ProviderException(error, code=response.status_code)

        data = response.json()
        if data.get("error"):
            raise ProviderException(data.get("error"), code=response.status_code)
        if data["status"] == "succeeded":
            original_result = data["analyzeResult"]
            standardized_response = microsoft_ocr_async_standardize_response(
                original_result
            )
            return AsyncResponseType[OcrAsyncDataClass](
                original_response=data,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        return AsyncPendingResponseType[OcrAsyncDataClass](
            provider_job_id=provider_job_id
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
            try:
                document_analysis_client = DocumentAnalysisClient(
                    endpoint=self.url["documentintelligence"],
                    credential=AzureKeyCredential(
                        self.api_settings["documentintelligence"]["subscription_key"]
                    ),
                )
                document_type_value = (
                    "prebuilt-receipt"
                    if document_type == FinancialParserType.RECEIPT.value
                    else "prebuilt-invoice"
                )
                poller = document_analysis_client.begin_analyze_document(
                    document_type_value, file_
                )
                form_pages = poller.result()
            except AzureError as provider_call_exception:
                raise ProviderException(str(provider_call_exception))

            try:
                if form_pages is None or not hasattr(form_pages, "to_dict"):
                    raise AttributeError
                # AttributeError sometimes happens in the lib when calling to dict
                # and a DocumentField has a None value
                original_response = form_pages.to_dict()
            except AttributeError:
                raise ProviderException("Provider return an empty response")
        standardized_response = microsoft_financial_parser_formatter(original_response)
        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
