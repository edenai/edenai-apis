import enum
import json
import re
from typing import List, Sequence
from typing import Tuple
from http import HTTPStatus
import requests

import google
import google.auth
import googleapiclient.discovery
from google.cloud.documentai_v1beta3 import Document
from google.api_core.exceptions import GoogleAPIError
from google.oauth2 import service_account

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
    Page,
    Row,
    Table,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentEnum,
)
from edenai_apis.utils.conversion import convert_pitch_from_percentage_to_semitones
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.conversion import convert_string_to_number
from edenai_apis.utils.types import AsyncResponseType
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
    FinancialMerchantInformation,
    FinancialCustomerInformation,
    FinancialBankInformation,
    FinancialLocalInformation,
    FinancialPaymentInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialParserObjectDataClass,
)


class GoogleVideoFeatures(enum.Enum):
    LABEL = "LABEL"
    TEXT = "TEXT"
    FACE = "FACE"
    PERSON = "PERSON"
    LOGO = "LOGO"
    OBJECT = "OBJECT"
    EXPLICIT = "EXPLICIT"


def handle_google_call(function_to_call, **kwargs):
    error_encoding_str = "bad encoding"
    msg_exception_encoding = "Could not decode audio file, bad file encoding"

    wrong_job_id_strs = [
        "Unrecognized long running operation name",
        "Operation not found",
        "Invalid operation id",
    ]

    try:
        response = function_to_call(**kwargs)
    except GoogleAPIError as exc:
        try:
            status_code = exc.code
            if isinstance(status_code, HTTPStatus):
                status_code = status_code.value
            if not isinstance(status_code, int):
                try:
                    status_code = int(status_code)
                except:
                    status_code = None
        except:
            status_code = None
        try:
            message = exc.message
        except:
            message = str(exc)
        if error_encoding_str in str(exc):
            message = msg_exception_encoding
        if any(str_error in str(exc) for str_error in wrong_job_id_strs):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID, code=status_code
            )
        raise ProviderException(message, code=status_code)
    except Exception as exc:
        message = str(exc)
        if any(str_error in str(exc) for str_error in message):
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)
        if error_encoding_str in message:
            message = msg_exception_encoding
        raise ProviderException(message)

    return response


def score_to_rate(score):
    return abs(score)


def google_video_get_job(provider_job_id: str):
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, _ = google.auth.default(scopes=scopes)
    service = googleapiclient.discovery.build(
        serviceName="videointelligence",
        version="v1",
        credentials=credentials,
        client_options={
            "api_endpoint": "https://videointelligence.googleapis.com/",
        },
    )
    payload_request = {"name": provider_job_id}
    request = handle_google_call(
        service.projects().locations().operations().get, **payload_request
    )

    result = handle_google_call(request.execute)

    return result


def score_to_sentiment(score: float) -> str:
    if score > 0:
        return SentimentEnum.POSITIVE.value
    elif score < 0:
        return SentimentEnum.NEGATIVE.value
    return SentimentEnum.NEUTRAL.value


# _Transform score of confidence to level of confidence
def score_to_content(score):
    if score == "UNKNOW":
        return 0
    elif score == "VERY_UNLIKELY":
        return 1
    elif score == "UNLIKELY":
        return 2
    elif score == "POSSIBLE":
        return 3
    elif score == "LIKELY":
        return 4
    elif score == "VERY_LIKELY":
        return 5
    else:
        return 0


def google_ocr_tables_standardize_response(
    original_response: dict,
) -> OcrTablesAsyncDataClass:
    """Standardize ocr table with dataclass from given google response"""
    try:
        raw_text: str = original_response["text"]
    except KeyError:
        raise ProviderException("Provider returned an empty response", 400)
    pages = [
        _ocr_tables_standardize_page(page, raw_text)
        for page in original_response.get("pages", [])
    ]

    return OcrTablesAsyncDataClass(
        pages=pages, num_pages=len(original_response["pages"])
    )


def _ocr_tables_standardize_page(page: dict, raw_text: str) -> Page:
    """Standardize one Page of a google ocr table response"""
    tables = [
        _ocr_tables_standardize_table(table, raw_text)
        for table in page.get("tables", [])
    ]
    return Page(tables=tables)


def _ocr_tables_standardize_table(table: dict, raw_text: str) -> Table:
    """Standardize one Table of a Page in a google ocr table response"""
    ocr_num_cols = 0
    rows: Sequence[Row] = []
    for row in table.get("headerRows", []):
        row, num_row_cols = _ocr_tables_standardize_row(row, raw_text, is_header=True)
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    for row in table.get("bodyRows", []):
        row, num_row_cols = _ocr_tables_standardize_row(row, raw_text)
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    return Table(rows=rows, num_rows=len(rows), num_cols=ocr_num_cols)


def _ocr_tables_standardize_row(row, raw_text, is_header=False) -> Tuple[Row, int]:
    """Standardize one Row of a Table in a Page of google ocr table response
    Returns:
        [Row, int]: the Row object + the num of cells included in the row
    """
    cells: Sequence[Cell] = []
    for cell in row.get("cells", []):
        std_cell = _ocr_tables_standardize_cell(cell, raw_text, is_header)
        cells.append(std_cell)
    ocr_row = Row(cells=cells)
    return ocr_row, len(cells)


def _ocr_tables_standardize_cell(cell, raw_text, is_header) -> Cell:
    """Standardize one Cell of a Row in a Table in a Page of google ocr table response"""
    vertices = cell["layout"]["boundingPoly"]["normalizedVertices"]
    text = ""
    for segment in cell["layout"]["textAnchor"].get("textSegments", []):
        start_index = int(segment.get("startIndex", 0))
        end_index = int(segment.get("endIndex", 0))
        text += raw_text[start_index:end_index]

    return Cell(
        text=text,
        col_index=None,
        row_index=None,
        row_span=cell["rowSpan"],
        col_span=cell["colSpan"],
        confidence=cell["layout"]["confidence"],
        is_header=is_header,
        bounding_box=BoundixBoxOCRTable(
            left=float(vertices[0].get("x", 0)),
            top=float(vertices[0].get("y", 0)),
            width=float(vertices[2].get("x", 0) - vertices[0].get("x", 0)),
            height=float(vertices[2].get("y", 0) - vertices[0].get("y", 0)),
        ),
    )


def get_tag_name(tag):
    """
    Get name of syntax tag of provider selected
    :param provider:     String that contains the name of provider
    :param tag:          String that contains the acronym of syntax tag
    :return:             String that contains the name of syntax tag
    """
    return {
        "ADJ": "Adjactive",
        "ADP": "Adposition",
        "ADV": "Adverb",
        "AFFIX": "Affix",
        "CONJ": "Coordinating_Conjunction",
        "DET": "Determiner",
        "NOUN": "Noun",
        "NUM": "Cardinal_number",
        "PRON": "Pronoun",
        "PRT": "Particle",
        "PUNCT": "Punctuation",
        "VERB": "Verb",
        "X": "Other",
    }[tag]


# *****************************Speech to text***************************************************
def get_encoding_and_sample_rate(extension: str):
    list_encoding = [
        ("LINEAR16", None),
        ("MULAW", None),
        ("AMR", 8000),
        ("AMR_WB", 16000),
        ("OGG_OPUS", 24000),
        ("SPEEX_WITH_HEADER_BYTE", 16000),
        ("WEBM_OPUS", 24000),
    ]
    if extension in ["wav", "flac"]:
        return None, None
    if extension.startswith("mp"):
        return "ENCODING_UNSPECIFIED", None
    if extension == "l16":
        extension = "linear16"
    if extension == "spx":
        extension = "speex"
    if "-" in extension:
        extension.replace("-", "_")
    right_encoding_sample: Tuple = next(
        filter(lambda x: extension in x[0].lower(), list_encoding), (None, None)
    )
    return right_encoding_sample


# *****************************Text to speech**************************************************#
def get_formated_speaking_rate(speaking_rate: int):
    if speaking_rate > 100:
        speaking_rate = 100
    if speaking_rate < -100:
        speaking_rate = -100
    if speaking_rate >= 0:
        diff = speaking_rate / 100
        return 1 + diff
    diff = -1 / 2 * speaking_rate / 100
    return 1 - diff


def get_formated_speaking_volume(speaking_volume: int):
    if speaking_volume > 100:
        speaking_volume = 100
    if speaking_volume < -100:
        speaking_volume = -100
    return speaking_volume * 6 / 100


def generate_tts_params(speaking_rate, speaking_pitch, speaking_volume):
    attribs = {
        "speaking_rate": (speaking_rate, get_formated_speaking_rate(speaking_rate)),
        "pitch": (
            speaking_pitch,
            convert_pitch_from_percentage_to_semitones(speaking_pitch),
        ),
        "volume_gain_db": (
            speaking_volume,
            get_formated_speaking_volume(speaking_volume),
        ),
    }
    params = {}

    for k, v in attribs.items():
        if not v[0]:
            continue
        params[k] = v[1]
    return params


def get_right_audio_support_and_sampling_rate(
    audio_format: str, list_audio_formats: List
) -> Tuple[str, str]:
    extension = audio_format
    if not audio_format:
        audio_format = "mp3"
    if audio_format == "wav":
        audio_format = "wav-linear16"
    extension = audio_format
    if "-" in audio_format:
        extension, audio_format = audio_format.split("-")
    right_audio_format = next(
        filter(lambda x: audio_format in x.lower(), list_audio_formats), None
    )
    return extension, right_audio_format or audio_format


def handle_done_response_ocr_async(
    result, client, job_id
) -> AsyncResponseType[OcrAsyncDataClass]:
    gcs_destination_uri = result["response"]["responses"][0]["outputConfig"][
        "gcsDestination"
    ]["uri"]
    match = re.match(r"gs://([^/]+)/(.+)", gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = client.get_bucket(bucket_name)

    blob_list = [
        blob
        for blob in list(bucket.list_blobs(prefix=prefix))
        if not blob.name.endswith("/")
    ]

    original_response = {"responses": []}
    pages: List[Page] = []
    for blob in blob_list:
        output = blob

        json_string = output.download_as_bytes()
        response = json.loads(json_string)

        for response in response["responses"]:
            original_response["responses"].append(response["fullTextAnnotation"])
            for page in response["fullTextAnnotation"]["pages"]:
                lines: Sequence[Line] = []
                for block in page["blocks"]:
                    words: Sequence[Word] = []
                    for paragraph in block["paragraphs"]:
                        line_boxes = BoundingBox.from_normalized_vertices(
                            paragraph["boundingBox"]["normalizedVertices"]
                        )
                        for word in paragraph["words"]:
                            word_boxes = BoundingBox.from_normalized_vertices(
                                word["boundingBox"]["normalizedVertices"]
                            )
                            word_text = ""
                            for symbol in word["symbols"]:
                                word_text += symbol["text"]
                            words.append(
                                Word(
                                    text=word_text,
                                    bounding_box=word_boxes,
                                    confidence=word["confidence"],
                                )
                            )
                lines.append(
                    Line(
                        text=" ".join([word.text for word in words]),
                        words=words,
                        bounding_box=line_boxes,
                        confidence=paragraph["confidence"],
                    )
                )
        pages.append(OcrAsyncPage(lines=lines))

    raw_text = "".join([res["text"] for res in original_response["responses"]])
    return AsyncResponseType(
        provider_job_id=job_id,
        original_response=original_response,
        standardized_response=OcrAsyncDataClass(
            raw_text=raw_text, pages=pages, number_of_pages=len(pages)
        ),
    )


def get_access_token(location: str):
    """
    Retrieves an access token for the Google Cloud Platform using service account credentials.

    Args:
        location (str): The file location of the service account credentials.

    Returns:
        str: The access token required for making API REST calls.

    Example:
        location = "/path/to/credentials.json"
        access_token = get_access_token(location)
        # Use the access_token for API REST calls
        response = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})

    """
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        location, scopes=scopes
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


# *****************************Financial Parser***************************************************
def format_document_to_dict(document: Document) -> List[dict]:
    """
    Organize the response from a Google document parser into a more structured output.
    Each element in the list represents a page of the document (e.g., invoice or receipt) with its fields.

    Args:
    - document (Document): The parsed Google document.

    Returns:
    - List[Dict]: A list of dictionaries, each containing organized information about a document page.
    """
    extracted_data = []

    for idx in range(0, len(Document.to_dict(document).get("pages"))):
        summary = {"line_items": []}
        for entity in document.entities:
            entity_dict = Document.Entity.to_dict(entity)
            page_anchor = entity_dict.get("page_anchor", {}) or {}
            page_refs = page_anchor.get("page_refs", [{}]) or [{}]
            if page_refs[0].get("page") != str(idx):
                continue
            type = entity_dict["type_"]
            if type == "line_item":
                line_dict = {}
                for property in entity_dict.get("properties", []):
                    property_type = property.get("type_", "")
                    property_value = property.get("normalized_value", {}).get(
                        "text"
                    ) or property.get("mention_text")
                    line_dict.update({property_type: property_value})
                summary["line_items"].append(line_dict)
            else:
                summary[type] = entity_dict.get("normalized_value", {}).get(
                    "text"
                ) or entity_dict.get("mention_text")
        summary["metadata"] = {"page_number": idx + 1, "invoice": idx + 1}
        extracted_data.append(summary)

    return extracted_data


def google_financial_parser(document: Document) -> FinancialParserDataClass:
    """
    Parse a Google document using a financial parser and return organized financial data.

    Args:
    - document (Document): The Google document to be parsed.

    Returns:
    - FinancialParserDataClass: Parsed financial data organized into a data class.
    """
    formatted_response = format_document_to_dict(document=document)
    extracted_data = []

    for page_document in formatted_response:
        extracted_data.append(
            FinancialParserObjectDataClass(
                # Customer Information
                customer_information=FinancialCustomerInformation(
                    name=page_document.get("receiver_name"),
                    shipping_address=page_document.get("ship_to_address"),
                    remittance_address=page_document.get("remit_to_address"),
                    billing_address=page_document.get("receiver_address"),
                    email=page_document.get("receiver_email"),
                    tax_id=page_document.get("receiver_tax_id"),
                ),
                # Merchant Information
                merchant_information=FinancialMerchantInformation(
                    website=page_document.get("supplier_website"),
                    tax_id=page_document.get("supplier_tax_id"),
                    address=page_document.get("supplier_address"),
                    phone=page_document.get("supplier_phone"),
                    email=page_document.get("supplier_email"),
                    name=page_document.get("supplier_name"),
                    business_number=page_document.get("supplier_registration"),
                ),
                # Payment Information
                payment_information=FinancialPaymentInformation(
                    total=convert_string_to_number(
                        page_document.get("total_amount"), float
                    ),
                    amount_due=convert_string_to_number(
                        page_document.get("net_amount"), float
                    ),
                    total_tax=convert_string_to_number(
                        page_document.get("total_tax_amount"), float
                    ),
                    payment_terms=page_document.get("payment_terms"),
                    prior_balance=convert_string_to_number(
                        page_document.get("amount_paid_since_last_invoice"), float
                    ),
                    amount_shipping=convert_string_to_number(
                        page_document.get("freight_amount"), float
                    ),
                    payment_method=page_document.get("payment_type"),
                    payment_card_number=page_document.get(
                        "credit_card_last_four_digits"
                    ),
                ),
                # Financial Document Information
                financial_document_information=FinancialDocumentInformation(
                    invoice_receipt_id=page_document.get("invoice_id"),
                    invoice_number=page_document.get("invoice_number"),
                    purchase_order=page_document.get("purchase_order"),
                    time=page_document.get("purchase_time"),
                    invoice_date=page_document.get("invoice_date")
                    or page_document.get("receipt_date"),
                    invoice_due_date=page_document.get("due_date"),
                    service_end_date=page_document.get("delivery_date"),
                ),
                # Bank Information
                bank=FinancialBankInformation(iban=page_document.get("supplier_iban")),
                # Local Information
                local=FinancialLocalInformation(
                    currency_exchange_rate=page_document.get("currency_exchange_rate"),
                    currency=page_document.get("currency"),
                ),
                # Item Lines
                item_lines=[
                    FinancialLineItem(
                        description=item.get("line_item/description"),
                        unit_price=convert_string_to_number(
                            item.get("line_item/unit_price"), float
                        ),
                        quantity=convert_string_to_number(
                            item.get("line_item/quantity"), int
                        ),
                        amount_line=convert_string_to_number(
                            item.get("line_item/amount"), float
                        ),
                    )
                    for item in page_document.get("line_items", [])
                ],
                # Invoice Metadata
                document_metadata=FinancialDocumentMetadata(
                    document_page_number=page_document["metadata"]["page_number"],
                    document_type=page_document.get("invoice_type"),
                ),
            )
        )

    return FinancialParserDataClass(extracted_data=extracted_data)


# ************************************* Chat ****************************************


def calculate_usage_tokens(original_response: dict) -> dict:
    """
    Calculates the token usage from the original response.
    """
    original_response["usage"] = {
        "prompt_tokens": original_response["usageMetadata"].get("promptTokenCount", 0),
        "completion_tokens": original_response["usageMetadata"].get(
            "candidatesTokenCount", 0
        ),
        "total_tokens": original_response["usageMetadata"].get("totalTokenCount", 0),
    }


def gemini_request(payload: dict, model: str, api_key: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    response = requests.post(url, json=payload)
    try:
        original_response = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderException(
            "An error occurred while parsing the response."
        ) from exc

    if response.status_code != 200:
        raise ProviderException(
            message=original_response["error"]["message"],
            code=response.status_code,
        )
    # calculate_usage_tokens(original_response=original_response)
    return original_response


def palm_request(payload: dict, model: str, location: str, token: str, project_id: str):
    url_subdomain = "us-central1-aiplatform"
    location = "us-central1"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:predict"

    response = requests.post(url=url, headers=headers, json=payload)
    try:
        original_response = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderException("An error occured while parsing the response.") from exc

    if response.status_code != 200:
        raise ProviderException(
            message=original_response["error"]["message"], code=response.status_code
        )
    return original_response
