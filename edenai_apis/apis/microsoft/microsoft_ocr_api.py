from collections import defaultdict
from io import BufferedReader
from typing import Sequence

import requests
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from edenai_apis.apis.microsoft.microsoft_helpers import normalize_invoice_result
from edenai_apis.features.ocr import (
    Bounding_box,
    IdentityParserDataClass,
    InfosIdentityParserDataClass,
    InfosReceiptParserDataClass,
    InvoiceParserDataClass,
    ItemLines,
    MerchantInformation,
    OcrDataClass,
    PaymentInformation,
    ReceiptParserDataClass,
    Taxes,
    get_info_country,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    InfoCountry,
    ItemIdentityParserDataClass,
    format_date,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.utils.conversion import add_query_param_in_url
from edenai_apis.utils.exception import AsyncJobException, AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from PIL import Image as Img


class MicrosoftOcrApi(OcrInterface):
    def ocr__ocr(
        self, 
        file: str, 
        language: str,
        file_url: str= "",
    ) -> ResponseType[OcrDataClass]:

        with open(file, "rb") as file_:
            file_content = file_.read()

        url = f"{self.api_settings['vision']['url']}/ocr?detectOrientation=true"

        response = requests.post(
            url=add_query_param_in_url(url, {"language": language}),
            headers=self.headers["vision"],
            data=file_content,
        ).json()

        final_text = ""

        if "error" in response:
            raise ProviderException(response["error"]["message"])

        # Get width and hight
        width, height = Img.open(file).size

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
        self, 
        file: str, 
        language: str,
        file_url: str= ""
    ) -> ResponseType[InvoiceParserDataClass]:

        file_ = open(file, "rb")
        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.api_settings["form_recognizer"]["url"],
                credential=AzureKeyCredential(
                    self.api_settings["form_recognizer"]["subscription_key"]
                ),
            )
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-invoice", file_
            )
            invoices = poller.result()
        except AzureError as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        original_response = invoices.to_dict()
        file_.close()

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=normalize_invoice_result(original_response),
        )

    def ocr__receipt_parser(
        self, 
        file: str, 
        language: str,
        file_url: str= ""
    ) -> ResponseType[ReceiptParserDataClass]:

        file_ = open(file, "rb")
        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.api_settings["form_recognizer"]["url"],
                credential=AzureKeyCredential(
                    self.api_settings["form_recognizer"]["subscription_key"]
                ),
            )
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-receipt", file_
            )
            form_pages = poller.result()
        except AzureError as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        original_response = form_pages.to_dict()
        file_.close()

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
            )

            # 5. Taxes
            taxes = [Taxes(taxes=fields.get("Tax", default_dict).get("value"))]

            # 6. Receipt infos / payment informations
            receipt_infos = {"doc_type": fields.get("doc_type")}
            payment_infos = PaymentInformation(
                tip=fields.get("Tip", default_dict).get("value")
            )

            # 7. Items
            items = []
            for item in fields.get("Items", default_dict).get("value", []):
                description = item["value"].get("Name", default_dict).get("value")
                price = item["value"].get("Price", default_dict).get("value")
                quantity_str = item["value"].get("Quantity", default_dict).get("value")
                quantity = int(quantity_str) if quantity_str else None
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
        self, 
        file: str, 
        file_url: str= ""
    ) -> ResponseType[IdentityParserDataClass]:

        file_ = open(file, "rb")
        try:
            document_analysis_client = DocumentAnalysisClient(
                endpoint=self.api_settings["form_recognizer"]["url"],
                credential=AzureKeyCredential(
                    self.api_settings["form_recognizer"]["subscription_key"]
                ),
            )
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-idDocument", file_
            )
            response = poller.result()
        except AzureError as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        original_response = response.to_dict()

        file_.close()

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
                    country=country,
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
                )
            )

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__ocr_tables_async__launch_job(
        self, 
        file: str, 
        file_type: str, 
        language: str,
        file_url: str= ""
    ) -> AsyncLaunchJobResponseType:

        with open(file, "rb") as file_:
            file_content = file_.read()
        url = (
            f"{self.api_settings['form_recognizer']['url']}formrecognizer/documentModels/"
            f"prebuilt-layout:analyze?api-version=2022-08-31"
        )
        url = add_query_param_in_url(url, {"locale": language})

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": self.api_settings["form_recognizer"][
                    "subscription_key"
                ],
            },
            data=file_content,
        )

        if response.status_code != 202:
            error = response.json()["error"]["innerror"]["message"]
            raise ProviderException(error)

        return AsyncLaunchJobResponseType(
            provider_job_id=response.headers.get("apim-request-id")
        )

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> AsyncBaseResponseType[OcrTablesAsyncDataClass]:

        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.api_settings["form_recognizer"][
                "subscription_key"
            ],
        }

        url = (
            self.api_settings["form_recognizer"]["url"]
            + f"formrecognizer/documentModels/prebuilt-layout/"
            f"analyzeResults/{job_id}?api-version=2022-08-31"
        )
        response = requests.get(url, headers=headers)

        if response.status_code >= 400:
            error = response.json()["error"]["message"]
            if "Resource not found" in error:
                raise AsyncJobException(
                    reason = AsyncJobExceptionReason.DEPRECATED_JOB_ID
                )
            raise ProviderException(error)

        data = response.json()
        if data.get("error"):
            raise ProviderException(data.get("error"))
        if data["status"] == "succeeded":
            original_result = data["analyzeResult"]

            def microsoft_async(original_result):
                page_num = 1
                pages: Sequence[Page] = []
                num_pages = len(original_result["pages"])
                tables: Sequence[Table] = []
                for table in original_result["tables"]:
                    rows: Sequence[Row] = []
                    num_rows = table["rowCount"]
                    num_cols = table["columnCount"]
                    row_index = 0
                    cells: Sequence[Cell] = []
                    all_header = True  # all cells in a row are "header" kind
                    is_header = False
                    for cell in table["cells"]:
                        bounding_box = cell["boundingRegions"][0]["polygon"]
                        width = original_result["pages"][page_num - 1]["width"]
                        height = original_result["pages"][page_num - 1]["height"]

                        ocr_cell = Cell(
                            text=cell["content"],
                            row_span=cell.get("rowSpan", 1),
                            col_span=cell.get("columnSpan", 1),
                            bounding_box=BoundixBoxOCRTable(
                                height=(bounding_box[7] - bounding_box[3]) / height,
                                width=(bounding_box[2] - bounding_box[0]) / width,
                                left=bounding_box[1] / width,
                                top=bounding_box[0] / height,
                            ),
                        )
                        if (
                            "kind" not in cell.keys()
                            or "kind" in cell.keys()
                            and "Header" not in cell["kind"]
                        ):
                            all_header = False
                        current_row_index = cell["rowIndex"]
                        if current_row_index > row_index:
                            row_index = current_row_index
                            if all_header:
                                is_header = True
                            all_header = True
                            rows.append(Row(is_header=is_header, cells=cells))
                            cells = []
                        cells.append(ocr_cell)
                        current_page_num = cell["boundingRegions"][0]["pageNumber"]
                        if current_page_num > page_num:
                            page_num = current_page_num
                            pages.append(Page())
                            tables = []
                    ocr_table = Table(rows=rows, num_cols=num_cols, num_rows=num_rows)
                    tables.append(ocr_table)
                    ocr_page = Page(tables=tables)
                try:
                    pages.append(ocr_page)
                except UnboundLocalError:
                    raise ProviderException("No table found in the document.")
                standardized_response = OcrTablesAsyncDataClass(
                    pages=pages, num_pages=num_pages
                )
                return standardized_response.dict()

            standardized_response = microsoft_async(original_result)
            return AsyncResponseType[OcrTablesAsyncDataClass](
                original_response=data,
                standardized_response=standardized_response,
                provider_job_id=job_id,
            )

        return AsyncPendingResponseType[OcrTablesAsyncDataClass](provider_job_id=job_id)
