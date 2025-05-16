import urllib
from pathlib import Path
from time import time
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Sequence

import requests
from botocore.exceptions import ClientError, ParamValidationError
from trp import Document

from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import (
    CustomDocumentParsingAsyncBoundingBox,
    CustomDocumentParsingAsyncDataClass,
    CustomDocumentParsingAsyncItem,
)
from edenai_apis.features.ocr.data_extraction.data_extraction_dataclass import (
    DataExtractionDataClass,
    ItemDataExtraction,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    InvoiceParserDataClass,
    InfosInvoiceParserDataClass,
    TaxesInvoice,
    LocaleInvoice,
    ItemLinesInvoice,
    MerchantInformationInvoice,
    CustomerInformationInvoice,
)
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import (
    BoundingBox,
    Line,
    OcrAsyncDataClass,
    Word,
    Page as OcrAsyncPage,
)
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page as OcrTablesPage,
    Row,
    Table,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
    InfosReceiptParserDataClass,
    MerchantInformation,
    CustomerInformation,
    Taxes,
    ItemLines,
    Locale,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialBankInformation,
    FinancialBarcode,
    FinancialCustomerInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialPaymentInformation,
)
from edenai_apis.features.video.explicit_content_detection_async.explicit_content_detection_async_dataclass import (
    ContentNSFW,
)
from edenai_apis.features.video.face_detection_async.face_detection_async_dataclass import (
    FaceAttributes,
    LandmarksVideo,
    VideoBoundingBox,
    VideoFace,
    VideoFacePoses,
)
from edenai_apis.features.video.label_detection_async.label_detection_async_dataclass import (
    VideoLabel,
    VideoLabelBoundingBox,
    VideoLabelTimeStamp,
)
from edenai_apis.features.video.person_tracking_async.person_tracking_async_dataclass import (
    PersonLandmarks,
    PersonTracking,
    VideoPersonPoses,
    VideoPersonQuality,
    VideoTrackingBoundingBox,
    VideoTrackingPerson,
)
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    VideoText,
    VideoTextBoundingBox,
    VideoTextFrames,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.bounding_box import BoundingBox as BBox
from edenai_apis.utils.conversion import convert_string_to_number
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.ssml import convert_audio_attr_in_prosody_tag
from edenai_apis.utils.types import (
    ResponseType,
)
from .config import storage_clients


def check_webhook_result(job_id: str, api_settings: dict) -> Dict:
    """Try get result on webhook.site with job id

    Args:
        job_id (str): async job id to get result to

    Returns:
        Dict: Result dict
    """
    webhook_token = api_settings["webhook_token"]
    api_key = api_settings["webhook_api_key"]
    webhook_get_url = (
        f"https://webhook.site/token/{webhook_token}/requests"
        + f"?sorting=newest&query={urllib.parse.quote_plus('content:'+str(job_id))}"
    )
    webhook_response = requests.get(url=webhook_get_url, headers={"Api-Key": api_key})
    response_status = webhook_response.status_code
    try:
        return webhook_response.json().get("data"), response_status
    except Exception:
        return None, response_status


def amazon_ocr_tables_parser(original_result) -> OcrTablesAsyncDataClass:
    document = Document(original_result)
    std_pages = [_ocr_tables_standarize_page(page) for page in document.pages]
    return OcrTablesAsyncDataClass(pages=std_pages, num_pages=len(std_pages))


def _ocr_tables_standarize_page(page) -> OcrTablesPage:
    std_tables = [_ocr_tables_standarize_table(table) for table in page.tables]
    return OcrTablesPage(tables=std_tables)


def _ocr_tables_standarize_table(table) -> Table:
    rows: Sequence[Row] = []
    num_cols = 0
    row_cols = 0
    for row in table.rows:
        std_row, row_cols = _ocr_tables_standarize_row(row)
        rows.append(std_row)
    # Since some cells are merged some row have less cols than others.
    # We chose to return the max number of cols
    num_cols = max(num_cols, row_cols)
    return Table(rows=rows, num_cols=num_cols, num_rows=len(rows))


def _ocr_tables_standarize_row(row) -> Tuple[Row, int]:
    """Returns Row as well has the number of colums in said row"""
    is_header = False
    cells: Sequence[Cell] = []
    for cell in row.cells:
        std_cell = _ocr_tables_standarize_cell(cell)
        cells.append(std_cell)

    num_col = len(cells)
    return Row(cells=cells, is_header=is_header), num_col


def _ocr_tables_standarize_cell(cell) -> Cell:
    is_header = "COLUMN_HEADER" in cell.entityTypes
    confidence = float(cell.confidence / 100)
    return Cell(
        text=cell.mergedText,
        row_index=cell.columnIndex,
        col_index=cell.rowIndex,
        row_span=cell.rowSpan,
        col_span=cell.columnSpan,
        confidence=confidence,
        is_header=is_header,
        bounding_box=BoundixBoxOCRTable(
            left=cell.geometry.boundingBox.left,
            top=cell.geometry.boundingBox.top,
            width=cell.geometry.boundingBox.width,
            height=cell.geometry.boundingBox.height,
        ),
    )


T = TypeVar("T")


# Video analysis async
def _upload_video_file_to_amazon_server(file: str, file_name: str, api_settings: Dict):
    """
    :param video:       String that contains the video file path
    :return:            String that contains the filename on the server
    """
    # Store file in an Amazon server
    file_extension = file.split(".")[-1]
    filename = str(int(time())) + file_name.stem + "_video_." + file_extension
    storage_clients(api_settings)["video"].meta.client.upload_file(
        file, api_settings["bucket_video"], filename
    )

    return filename


def amazon_get_video_data(file: str):
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    # Upload video to amazon server
    filename = _upload_video_file_to_amazon_server(file, Path(file), api_settings)

    # Get response
    role = api_settings["role"]
    topic = api_settings["topic_video"]
    bucket = api_settings["bucket_video"]
    video = {"S3Object": {"Bucket": bucket, "Name": filename}}
    notification_channel = {
        "RoleArn": role,
        "SNSTopicArn": topic,
    }
    return video, notification_channel


def amazon_video_original_response(
    job_id: str,
    max_result: int,
    next_token: str,
    function_to_call: Callable,
    sortBy: Optional[str] = None,
):
    params = {"JobId": job_id, "MaxResults": max_result, "NextToken": next_token}
    if sortBy:
        params.update({"SortBy": sortBy})

    response = handle_amazon_call(function_to_call, **params)
    return response


def amazon_custom_document_parsing_formatter(
    pages: List[dict],
) -> ResponseType[CustomDocumentParsingAsyncDataClass]:
    """
    Take response form amazon by pages
    Return custom document parser dataclass
    """
    items = []
    for index, page in enumerate(pages):
        for block in page["Blocks"]:
            if block["BlockType"] == "QUERY_RESULT":
                if block.get("Geometry"):
                    left = block["Geometry"]["BoundingBox"]["Left"]
                    top = block["Geometry"]["BoundingBox"]["Top"]
                    width = block["Geometry"]["BoundingBox"]["Width"]
                    height = block["Geometry"]["BoundingBox"]["Height"]
                else:
                    left, top, width, height = None, None, None, None
                bounding_box = CustomDocumentParsingAsyncBoundingBox(
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                )
                query = query_answer_result(page["Blocks"], block["Id"])
                if not query:
                    continue
                item = CustomDocumentParsingAsyncItem(
                    confidence=block["Confidence"],
                    value=block["Text"],
                    query=query,
                    page=block.get("Page", index + 1),
                    bounding_box=bounding_box,
                )
                items.append(item)
    return CustomDocumentParsingAsyncDataClass(items=items)


def query_answer_result(page: List[dict], identifier: str):
    """
    Retrieve the text of a query based on its relationship ID.

    Parameters:
        page (List[dict]): List of blocks, each representing a query or answer.
        identifier (str): The relationship ID to match against.

    Returns:
        str: The text of the query with a matching relationship ID. If no match is found, returns None.
    """
    queries = [q for q in page if q["BlockType"] == "QUERY"]
    for query in queries:
        relationships = query.get("Relationships")
        if relationships:
            for relationship in relationships:
                if identifier in relationship["Ids"]:
                    return query["Query"]["Text"]
    return None


def amazon_invoice_parser_formatter(pages: List[dict]) -> InvoiceParserDataClass:
    extracted_data = []
    for page in pages:
        if page.get("JobStatus") == "FAILED":
            raise ProviderException(
                page.get("StatusMessage", "Amazon returned a job status: FAILED")
            )
        for invoice in page.get("ExpenseDocuments") or []:
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
                            quantity=convert_string_to_number(
                                parsed_items.get("QUANTITY"), float
                            ),
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
                customer_id=None,
                customer_number=summary.get("CUSTOMER_NUMBER"),
                customer_tax_id=None,
                customer_mailing_address=None,
                customer_billing_address=None,
                customer_shipping_address=None,
                customer_service_address=None,
                customer_remittance_address=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
                vat_number=None,
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
                abn_number=None,
                gst_number=None,
                pan_number=None,
                vat_number=None,
            )

            invoice_currency = None
            if len(currencies) == 1:
                invoice_currency = list(currencies.keys())[0]
            # HACK in case multiple currencies are returned,
            # we get the one who appeared the most
            elif len(currencies) > 1:
                invoice_currency = max(currencies, key=currencies.get)
            locale = LocaleInvoice(currency=invoice_currency, language=None)

            taxes = [
                TaxesInvoice(
                    value=convert_string_to_number(summary.get("TAX"), float), rate=None
                )
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
                previous_unpaid_balance=convert_string_to_number(
                    summary.get("PRIOR_BALANCE"), float
                ),
                discount=convert_string_to_number(summary.get("DISCOUNT"), float),
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
                po_number=summary.get("PO_NUMBER"),
            )
            extracted_data.append(invoice_infos)
    return InvoiceParserDataClass(extracted_data=extracted_data)


def amazon_receipt_parser_formatter(pages: List[dict]) -> ReceiptParserDataClass:
    extracted_data = []
    for page in pages:
        for receipt in page.get("ExpenseDocuments") or []:
            # format response to be more easily parsable
            summary = {}
            currencies = {}
            for field in receipt.get("SummaryFields") or []:
                field_type = field["Type"]["Text"]
                summary[field_type] = field["ValueDetection"]["Text"]
                field_currency = field.get("Currency", {}).get("Code")
                if field_currency is not None:
                    if field_currency not in currencies:
                        currencies[field_currency] = 1
                    else:
                        currencies[field_currency] += 1

            item_lines = []
            for line_item_group in receipt.get("LineItemGroups") or []:
                for fields in line_item_group.get("LineItems") or []:
                    parsed_items = {
                        item["Type"]["Text"]: item["ValueDetection"]["Text"]
                        for item in fields["LineItemExpenseFields"]
                    }
                    item_lines.append(
                        ItemLines(
                            description=parsed_items.get("ITEM"),
                            quantity=convert_string_to_number(
                                parsed_items.get("QUANTITY"), float
                            ),
                            amount=convert_string_to_number(
                                parsed_items.get("PRICE"), float
                            ),
                            unit_price=convert_string_to_number(
                                parsed_items.get("UNIT_PRICE"), float
                            ),
                        )
                    )
            customer = CustomerInformation(
                customer_name=summary.get("RECEIVER_NAME", summary.get("NAME")),
            )

            merchant = MerchantInformation(
                merchant_name=summary.get("VENDOR_NAME"),
                merchant_address=summary.get("VENDOR_ADDRESS"),
                merchant_phone=summary.get("VENDOR_PHONE"),
                merchant_url=summary.get("VENDOR_URL"),
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
            locale = Locale(currency=invoice_currency, language=None, country=None)

            taxes = [
                Taxes(
                    taxes=convert_string_to_number(summary.get("TAX"), float), rate=None
                )
            ]

            receipt_infos = InfosReceiptParserDataClass(
                customer_information=customer,
                merchant_information=merchant,
                invoice_number=summary.get("INVOICE_RECEIPT_ID"),
                invoice_total=convert_string_to_number(summary.get("TOTAL"), float),
                invoice_subtotal=convert_string_to_number(
                    summary.get("SUBTOTAL"), float
                ),
                taxes=taxes,
                date=summary.get("ORDER_DATE", summary.get("INVOICE_RECEIPT_DATE")),
                due_date=summary.get("DUE_DATE"),
                locale=locale,
                item_lines=item_lines,
                category=None,
                time=None,
            )
            extracted_data.append(receipt_infos)
    return ReceiptParserDataClass(extracted_data=extracted_data)


def amazon_financial_parser_formatter(pages: List[dict]) -> FinancialParserDataClass:
    """
    Parse Amazon financial response into a data class response by organizing the response.

    Args:
    - pages (List[dict]): List of pages from the Amazon financial response.

    Returns:
    - FinancialParserDataClass: Parsed financial data organized into a data class.
    """
    extracted_data = []

    for page_idx, page in enumerate(pages):
        if page.get("JobStatus") == "FAILED":
            raise ProviderException(
                page.get("StatusMessage", "Amazon returned a job status: FAILED")
            )

        for invoice in page.get("ExpenseDocuments") or []:
            summary = {}
            currencies = {}
            invoice_index = invoice["ExpenseIndex"]

            # Parse summary fields
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

            # Parse line item groups
            for line_item_group in invoice["LineItemGroups"]:
                for fields in line_item_group["LineItems"]:
                    parsed_items = {
                        item["Type"]["Text"]: item["ValueDetection"]["Text"]
                        for item in fields["LineItemExpenseFields"]
                    }
                    item_lines.append(
                        FinancialLineItem(
                            amount_line=convert_string_to_number(
                                parsed_items.get("PRICE"), float
                            ),
                            description=parsed_items.get("ITEM"),
                            quantity=convert_string_to_number(
                                parsed_items.get("QUANTITY"), int
                            ),
                            unit_price=convert_string_to_number(
                                parsed_items.get("UNIT_PRICE"), float
                            ),
                            product_code=parsed_items.get("PRODUCT_CODE"),
                        )
                    )

            # Build FinancialCustomerInformation object
            customer = FinancialCustomerInformation(
                name=summary.get("RECEIVER_NAME") or summary.get("NAME"),
                id_reference=summary.get("ID_REFERENCE"),
                mailing_address=summary.get("RECEIVER_ADDRESS"),
                remittance_address=summary.get("ADDRESS"),
                phone=summary.get("RECEIVER_PHONE"),
                vat_number=summary.get("RECEIVER_VAT_NUMBER"),
                abn_number=summary.get("RECEIVER_ABN_NUMBER"),
                gst_number=summary.get("RECEIVER_GST_NUMBER"),
                pan_number=summary.get("RECEIVER_PAN_NUMBER"),
                customer_number=summary.get("CUSTOMER_NUMBER"),
                tax_id=summary.get("TAX_PAYER_ID"),
            )

            # Build FinancialMerchantInformation object
            merchant = FinancialMerchantInformation(
                name=summary.get("VENDOR_NAME"),
                address=summary.get("VENDOR_ADDRESS"),
                phone=summary.get("VENDOR_PHONE"),
                vat_number=summary.get("VENDOR_VAT_NUMBER"),
                abn_number=summary.get("VENDOR_ABN_NUMBER"),
                gst_number=summary.get("VENDOR_GST_NUMBER"),
                pan_number=summary.get("VENDOR_PAN_NUMBER"),
                website=summary.get("VENDOR_URL"),
                city=summary.get("CITY"),
                country=summary.get("COUNTRY"),
                province=summary.get("STATE"),
                zip_code=summary.get("ZIP_CODE"),
            )

            # Build FinancialPaymentInformation object
            payment = FinancialPaymentInformation(
                amount_due=convert_string_to_number(summary.get("AMOUNT_DUE"), float),
                amount_paid=convert_string_to_number(summary.get("AMOUNT_PAID"), float),
                total=convert_string_to_number(summary.get("TOTAL"), float),
                subtotal=convert_string_to_number(summary.get("SUB_TOTAL"), float),
                service_charge=convert_string_to_number(
                    summary.get("SERVICE_CHARGE"), float
                ),
                payment_terms=summary.get("PAYMENT_TERMS"),
                shipping_handling_charge=convert_string_to_number(
                    summary.get("SHIPPING_HANDLING_CHARGE"), float
                ),
                prior_balance=convert_string_to_number(
                    summary.get("PRIOR_BALANCE"), float
                ),
                gratuity=convert_string_to_number(summary.get("GRATUITY"), float),
                discount=convert_string_to_number(summary.get("DISCOUNT"), float),
                total_tax=convert_string_to_number(summary.get("TAX"), float),
            )

            # Build FinancialDocumentInformation object
            financial_document_information = FinancialDocumentInformation(
                invoice_receipt_id=summary.get("INVOICE_RECEIPT_ID"),
                purchase_order=summary.get("PO_NUMBER"),
                invoice_date=summary.get("INVOICE_RECEIPT_DATE"),
                invoice_due_date=summary.get("DUE_DATE"),
                order_date=summary.get("ORDER_DATE"),
            )

            invoice_currency = None

            # Determine invoice currency
            if len(currencies) == 1:
                invoice_currency = list(currencies.keys())[0]
            elif len(currencies) > 1:
                invoice_currency = max(currencies, key=currencies.get)

            # Build FinancialLocalInformation object
            local = FinancialLocalInformation(
                currency=invoice_currency,
            )

            # Build FinancialBankInformation object
            bank = FinancialBankInformation(
                account_number=summary.get("ACCOUNT_NUMBER"),
            )

            # Build FinancialDocumentMetadata object
            document_metadata = FinancialDocumentMetadata(
                document_index=invoice_index, document_page_number=page_idx + 1
            )

            # Build FinancialParserObjectDataClass object
            financial_document = FinancialParserObjectDataClass(
                customer_information=customer,
                merchant_information=merchant,
                payment_information=payment,
                financial_document_information=financial_document_information,
                local=local,
                bank=bank,
                item_lines=item_lines,
                document_metadata=document_metadata,
            )
            extracted_data.append(financial_document)

    return FinancialParserDataClass(extracted_data=extracted_data)


def amazon_speaking_rate_converter(speaking_rate: int):
    if speaking_rate < -80:
        speaking_rate = -80
    if speaking_rate > 100:
        speaking_rate = 100
    return speaking_rate + 100


def amazon_speaking_volume_adapter(speaking_volume: int):
    if speaking_volume < -100:
        speaking_volume = 100
    if speaking_volume > 100:
        speaking_volume = 100
    return speaking_volume * 6 / 100


def generate_right_ssml_text(
    text, speaking_rate, speaking_pitch, speaking_volume
) -> str:
    attribs = {
        "rate": (speaking_rate, f"{amazon_speaking_rate_converter(speaking_rate)}%"),
        "pitch": (speaking_pitch, f"{speaking_pitch}%"),
        "volume": (
            speaking_volume,
            f"{amazon_speaking_volume_adapter(speaking_volume)}dB",
        ),
    }
    cleaned_attribs_string = ""
    for k, v in attribs.items():
        if not v[0]:
            continue
        cleaned_attribs_string = f"{cleaned_attribs_string} {k}='{v[1]}'"
    if not cleaned_attribs_string.strip():
        return text
    return convert_audio_attr_in_prosody_tag(cleaned_attribs_string, text)


def get_right_audio_support_and_sampling_rate(audio_format: str, sampling_rate: int):
    samplings = [8000, 16000, 22050, 24000]
    pcm_sampling = [8000, 16000]
    returned_audio_format = audio_format
    if not returned_audio_format:
        returned_audio_format = "mp3"
        audio_format = "mp3"
    if returned_audio_format == "ogg":
        returned_audio_format = "ogg_vorbis"
    if not sampling_rate:
        return audio_format, returned_audio_format, None
    nearest_sampling = (
        min(samplings, key=lambda x: abs(x - sampling_rate))
        if returned_audio_format != "pcm"
        else min(pcm_sampling, key=lambda x: abs(x - sampling_rate))
    )
    return audio_format, returned_audio_format, nearest_sampling


def _convert_response_to_blocks_with_id(responses: list) -> dict:
    """
    Convert the blocks from the response to a dict with Id as key
    """

    blocks_dict = {}
    for response in responses:
        for block in response.get("Blocks", []) or []:
            blocks_dict[block["Id"]] = block
    return blocks_dict


def amazon_ocr_async_formatter(responses: list) -> OcrAsyncDataClass:
    """
    Format the response from the OCR API to be more easily parsable

    Args
        response: the response from the OCR API

    Returns
        OcrAsyncDataClass: the formatted response
    """
    blocks: dict = _convert_response_to_blocks_with_id(responses)

    pages: Sequence[OcrAsyncPage] = []
    for _, block in blocks.items():
        if block["BlockType"] != "PAGE":
            continue

        lines: Sequence[Line] = []
        for block_id in block.get("Relationships", [{}])[0].get("Ids", []):
            if blocks[block_id]["BlockType"] != "LINE":
                continue

            words: Sequence[Word] = []
            for word_id in blocks[block_id]["Relationships"][0]["Ids"]:
                if blocks[word_id]["BlockType"] != "WORD":
                    continue

                word = Word(
                    text=blocks[word_id]["Text"],
                    bounding_box=BoundingBox.from_json(
                        bounding_box=blocks[word_id]["Geometry"]["BoundingBox"],
                        modifiers=lambda x: x.title(),
                    ),
                    confidence=blocks[word_id]["Confidence"],
                )
                words.append(word)

            line = Line(
                text=blocks[block_id]["Text"],
                words=words,
                bounding_box=BoundingBox.from_json(
                    bounding_box=blocks[block_id]["Geometry"]["BoundingBox"],
                    modifiers=lambda x: x.title(),
                ),
                confidence=blocks[block_id]["Confidence"],
            )
            lines.append(line)

        page = OcrAsyncPage(lines=lines)
        pages.append(page)

    text = ""
    for page in pages:
        for line in page.lines:
            text += line.text + "\n"

    return OcrAsyncDataClass(raw_text=text, pages=pages, number_of_pages=len(pages))


def amazon_data_extraction_formatter(
    responses: List[dict],
) -> DataExtractionDataClass:
    """
    Format the response for OCR Document parsing to be more easily parsable

    Args
        responses: the responses from Textract.Client.analyse_document

    return DataExtractionDataClass
    """
    blocks = _convert_response_to_blocks_with_id(responses)
    items: Sequence[ItemDataExtraction] = []

    for _, block in blocks.items():
        if block["BlockType"] != "KEY_VALUE_SET":
            continue

        if block["EntityTypes"] != ["KEY"]:
            continue

        if len(block.get("Relationships", [])) < 2:
            continue

        item = {}
        try:
            for relation in block["Relationships"]:
                if relation["Type"] == "CHILD":
                    item["key"] = blocks[relation["Ids"][0]]["Text"]
                elif relation["Type"] == "VALUE":
                    value_id = relation["Ids"][0]
                    child = blocks[blocks[value_id]["Relationships"][0]["Ids"][0]]
                    item["value"] = child["Text"]
                    item["bounding_box"] = BBox.from_json(
                        child["Geometry"]["BoundingBox"],
                        modifiers=lambda x: x.title(),
                    )

                    item["confidence_score"] = child["Confidence"] / 100

            items.append(ItemDataExtraction(**item))
        except KeyError:
            continue

    return DataExtractionDataClass(fields=items)


def handle_amazon_call(func: Callable, **kwargs):
    job_id_strings_errors = [
        "InvalidJobIdException",
        "Request has invalid Job Id",
        "Could not find JobId",
        "job couldn't be found",
    ]
    try:
        response = func(**kwargs)
    except ClientError as exc:
        response_meta = exc.response.get("ResponseMetadata", {}) or {}
        status_code = response_meta.get("HTTPStatusCode", None)
        if any(str_error in str(exc) for str_error in job_id_strings_errors):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID, code=status_code
            )
        raise ProviderException(str(exc), code=status_code)
    except ParamValidationError as exc:
        raise ProviderException(str(exc), code=400) from exc
    except Exception as exc:
        if any(str_error in str(exc) for str_error in job_id_strings_errors):
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)
        raise ProviderException(str(exc))
    return response


def amazon_video_person_tracking_parser(response):
    # gather all persons with the same index :
    persons_index = {index["Person"]["Index"] for index in response["Persons"]}
    tracked_persons = []
    for index in persons_index:
        detected_persons = [
            item for item in response["Persons"] if item["Person"]["Index"] == index
        ]
        tracked_person = []
        for detected_person in detected_persons:
            if detected_person["Person"].get("BoundingBox"):
                offset = float(detected_person["Timestamp"] / 1000.0)
                bounding_box = detected_person.get("Person").get("BoundingBox")
                bounding_box = VideoTrackingBoundingBox(
                    top=bounding_box.get("Top", 0),
                    left=bounding_box.get("Left", 0),
                    height=bounding_box.get("Height", 0),
                    width=bounding_box.get("Width", 0),
                )
                face = detected_person["Person"].get("Face")
                # Get landmarks
                poses = VideoPersonPoses.default()
                landmarks = PersonLandmarks()
                quality = VideoPersonQuality.default()
                if face:
                    landmarks_dict = {}
                    for land in face.get("Landmarks", []):
                        landmarks_dict[land["Type"]] = [land["X"], land["Y"]]
                    landmarks = PersonLandmarks(
                        eye_left=landmarks_dict.get("eyeLeft", []),
                        eye_right=landmarks_dict.get("eyeRight", []),
                        nose=landmarks_dict.get("nose", []),
                        mouth_left=landmarks_dict.get("mouthLeft", []),
                        mouth_right=landmarks_dict.get("mouthRight", []),
                    )
                    poses = VideoPersonPoses(
                        roll=face.get("Pose").get("Roll"),
                        yaw=face.get("Pose").get("Yaw"),
                        pitch=face.get("Pose").get("Pitch"),
                    )
                    quality = VideoPersonQuality(
                        brightness=face.get("Quality").get("Brightness"),
                        sharpness=face.get("Quality").get("Sharpness"),
                    )

                tracked_person.append(
                    PersonTracking(
                        offset=offset,
                        bounding_box=bounding_box,
                        landmarks=landmarks,
                        poses=poses,
                        quality=quality,
                    )
                )
        if len(tracked_person) > 0:
            tracked_persons.append(VideoTrackingPerson(tracked=tracked_person))
    return tracked_persons


def amazon_video_labels_parser(response):
    labels = []
    for label in response["Labels"]:
        # Category
        parents = []
        for parent in label["Label"]["Parents"]:
            if parent["Name"]:
                parents.append(parent["Name"])

        # bounding boxes
        boxes = []
        for instance in label["Label"]["Instances"]:
            video_box = VideoLabelBoundingBox(
                top=instance["BoundingBox"].get("Top", 0),
                left=instance["BoundingBox"].get("Left", 0),
                width=instance["BoundingBox"].get("Width", 0),
                height=instance["BoundingBox"].get("Height", 0),
            )
            boxes.append(video_box)

        videolabel = VideoLabel(
            timestamp=[
                VideoLabelTimeStamp(start=float(label["Timestamp"]) / 1000.0, end=None)
            ],
            confidence=label["Label"].get("Confidence", 0) / 100,
            name=label["Label"]["Name"],
            category=parents,
            bounding_box=boxes,
        )
        labels.append(videolabel)
    return labels


def amazon_video_text_parser(response):
    text_video = []
    # Get unique values of detected text annotation
    detected_texts = {
        text["TextDetection"]["DetectedText"] for text in response["TextDetections"]
    }

    # For each unique value, get all the frames where it appears
    for text in detected_texts:
        annotations = [
            item
            for item in response["TextDetections"]
            if item["TextDetection"]["DetectedText"] == text
        ]
        frames = []
        for annotation in annotations:
            timestamp = float(annotation["Timestamp"]) / 1000.0
            confidence = round(annotation["TextDetection"]["Confidence"] / 100, 2)
            geometry = annotation["TextDetection"]["Geometry"]["BoundingBox"]
            bounding_box = VideoTextBoundingBox(
                top=geometry.get("Top", 0),
                left=geometry.get("Left", 0),
                width=geometry.get("Width", 0),
                height=geometry.get("Height", 0),
            )
            frame = VideoTextFrames(
                timestamp=timestamp,
                confidence=confidence,
                bounding_box=bounding_box,
            )
            frames.append(frame)

        video_text = VideoText(
            text=text,
            frames=frames,
        )
        text_video.append(video_text)

    return text_video


def amazon_video_face_parser(response):
    faces = []
    for face in response["Faces"]:
        # Time stamp
        offset = float(face["Timestamp"]) / 1000.0  # convert to seconds

        # Bounding box
        bounding_box = VideoBoundingBox(
            top=face["Face"]["BoundingBox"].get("Top", 0),
            left=face["Face"]["BoundingBox"].get("Left", 0),
            height=face["Face"]["BoundingBox"].get("Height", 0),
            width=face["Face"]["BoundingBox"].get("Width", 0),
        )

        # Attributes
        poses = VideoFacePoses(
            pitch=face["Face"]["Pose"].get("Pitch", 0) / 100,
            yawn=face["Face"]["Pose"].get("Yaw", 0) / 100,
            roll=face["Face"]["Pose"].get("Roll", 0) / 100,
        )
        attributes_video = FaceAttributes(
            pose=poses,
            brightness=face["Face"]["Quality"].get("Brightness", 0) / 100,
            sharpness=face["Face"]["Quality"].get("Sharpness", 0) / 100,
            headwear=None,
            frontal_gaze=None,
            eyes_visible=None,
            glasses=None,
            mouth_open=None,
            smiling=None,
        )

        # Landmarks
        landmarks_output = {}
        for land in face["Face"]["Landmarks"]:
            if land.get("Type") and land.get("X") and land.get("Y"):
                landmarks_output[land["Type"]] = [land["X"], land["Y"]]

        landmarks_video = LandmarksVideo(
            eye_left=landmarks_output.get("eyeLeft", []),
            eye_right=landmarks_output.get("eyeRight", []),
            mouth_left=landmarks_output.get("mouthLeft", []),
            mouth_right=landmarks_output.get("mouthRight", []),
            nose=landmarks_output.get("nose", []),
        )
        faces.append(
            VideoFace(
                offset=offset,
                attributes=attributes_video,
                landmarks=landmarks_video,
                bounding_box=bounding_box,
            )
        )
    return faces


def amazon_video_explicit_parser(response):
    moderated_content = []
    for label in response.get("ModerationLabels"):
        confidence = label.get("ModerationLabel").get("Confidence")
        timestamp = float(label.get("Timestamp")) / 1000.0  # convert to seconds
        if label.get("ModerationLabel").get("ParentName"):
            category = label.get("ModerationLabel").get("ParentName")
        else:
            category = label.get("ModerationLabel").get("Name")
        moderated_content.append(
            ContentNSFW(
                timestamp=timestamp,
                confidence=confidence / 100,
                category=category,
            )
        )
    return moderated_content


def get_confidence_if_true(face_data, attribute_key):
    """
    retrieve the confidence of a value if its true (face_detection)
    """
    attribute_info = face_data.get(attribute_key, {})
    if attribute_info.get("Value") is True:
        return attribute_info.get("Confidence", 0.0) / 100.0
    return None
