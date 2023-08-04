import json
import urllib
from io import BufferedReader
from time import time
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Sequence, Union
from pathlib import Path
import requests
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
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ReceiptParserDataClass,
    InfosReceiptParserDataClass,
    MerchantInformation,
    CustomerInformation,
    Taxes,
    ItemLines,
    Locale,
)
from trp import Document

from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page as OcrTablesPage,
    Row,
    Table,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import validate_audio_attribute_against_ssml_tags_use
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    LanguageException,
    ProviderException,
)
from edenai_apis.utils.ssml import convert_audio_attr_in_prosody_tag

from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)

from .config import clients, storage_clients
from edenai_apis.utils.conversion import convert_string_to_number
from edenai_apis.utils.bounding_box import BoundingBox as BBox

from botocore.exceptions import ClientError, ParamValidationError

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
    return Cell(
        text=cell.mergedText,
        row_index=cell.columnIndex,
        col_index=cell.rowIndex,
        row_span=cell.rowSpan,
        col_span=cell.columnSpan,
        confidence=cell.confidence,
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


def amazon_launch_video_job(file: str, feature: str):
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    # Upload video to amazon server
    filename = _upload_video_file_to_amazon_server(file, Path(file), api_settings)

    # Get response
    role = api_settings["role"]
    topic = api_settings["topic_video"]
    bucket = api_settings["bucket_video"]

    features = {
        "LABEL": clients(api_settings)["video"].start_label_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "TEXT": clients(api_settings)["video"].start_text_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "FACE": clients(api_settings)["video"].start_face_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "PERSON": clients(api_settings)["video"].start_person_tracking(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "EXPLICIT": clients(api_settings)["video"].start_content_moderation(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
    }
    response = features.get(feature)
    # return job id
    job_id = response["JobId"]
    return job_id


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


def amazon_video_response_formatter(
    response: Dict, standardized_response: T, provider_job_id: str
) -> AsyncBaseResponseType[T]:
    if response["JobStatus"] == "SUCCEEDED":
        return AsyncResponseType[T](
            original_response=response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )
    elif response["JobStatus"] == "IN_PROGRESS":
        return AsyncPendingResponseType[T](provider_job_id=provider_job_id)
    elif response["JobStatus"] == "FAILED":
        error: Optional[str] = response.get("StatusMessage")
        raise ProviderException(error)
    raise ProviderException("Amazon did not return a JobStatus")


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
            first_relation_id = relationships[0]["Ids"][0]
            if first_relation_id == identifier:
                return query["Query"]["Text"]
    return None


def amazon_invoice_parser_formatter(pages: List[dict]) -> InvoiceParserDataClass:
    extracted_data = []
    for page in pages:
        if page.get("JobStatus") == "FAILED":
            raise ProviderException(
                page.get("StatusMessage", "Amazon returned a job status: FAILED")
            )
        for invoice in page["ExpenseDocuments"]:
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
        for receipt in page["ExpenseDocuments"]:
            # format response to be more easily parsable
            summary = {}
            currencies = {}
            for field in receipt["SummaryFields"]:
                field_type = field["Type"]["Text"]
                summary[field_type] = field["ValueDetection"]["Text"]
                field_currency = field.get("Currency", {}).get("Code")
                if field_currency is not None:
                    if field_currency not in currencies:
                        currencies[field_currency] = 1
                    else:
                        currencies[field_currency] += 1

            item_lines = []
            for line_item_group in receipt["LineItemGroups"]:
                for fields in line_item_group["LineItems"]:
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
            locale = Locale(
                currency=invoice_currency, invoice_language=None, country=None
            )

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


def generate_right_ssml_text(text, speaking_rate, speaking_pitch, speaking_volume) -> str:
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
        for block in response["Blocks"]:
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
        for block_id in block["Relationships"][0]["Ids"]:
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


def handle_amazon_call(func : Callable, **kwargs):
    job_id_strings_errors = [
        "InvalidJobIdException", 
        "Request has invalid Job Id", 
        "Could not find JobId", 
        "job couldn't be found"
    ]
    try:
        response = func(**kwargs)
    except ClientError as exc:
        response_meta = exc.response.get("ResponseMetadata", {}) or {}
        status_code = response_meta.get("HTTPStatusCode", None)
        if any(str_error in str(exc) for str_error in job_id_strings_errors):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                code = status_code
            )
        raise ProviderException(str(exc), code=status_code)
    except ParamValidationError as exc:    
        raise ProviderException(str(exc), code= 400) from exc
    except Exception as exc:
        if any(str_error in str(exc) for str_error in job_id_strings_errors):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID
            )
        raise ProviderException(str(exc))
    return response
