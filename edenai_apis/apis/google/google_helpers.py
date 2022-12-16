import copy
from typing import Sequence
from typing import Tuple

from google.oauth2.utils import enum
import google.auth
import google
import googleapiclient.discovery

from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)


class GoogleVideoFeatures(enum.Enum):
    LABEL = "LABEL"
    TEXT = "TEXT"
    FACE = "FACE"
    PERSON = "PERSON"
    LOGO = "LOGO"
    OBJECT = "OBJECT"
    EXPLICIT = "EXPLICIT"


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
    request = service.projects().locations().operations().get(name=provider_job_id)
    result = request.execute()
    return result


def score_to_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"


# _Transform score of confidence to level of confidence


def score_to_content(score):
    if score == "VERY_UNLIKELY":
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


def google_ocr_tables_standardize_response(original_response) -> OcrTablesAsyncDataClass:
    raw_text = original_response["text"]
    pages = [
        _ocr_tables_standardize_page(page, raw_text)
        for page in original_response.get("pages", [])
    ]

    return OcrTablesAsyncDataClass(
        pages=pages, num_pages=len(original_response["pages"])
    )


def _ocr_tables_standardize_page(page, raw_text) -> Page:
    tables = [
        _ocr_tables_standardize_table(table, raw_text)
        for table in page.get("tables", [])
    ]
    return Page(tables=tables)


def _ocr_tables_standardize_table(table, raw_text) -> Table:
    ocr_num_cols = 0
    rows: Sequence[Row] = []
    for row, row_index in table.get("headerRows", []):
        row, num_row_cols = _ocr_tables_standardize_row(
            row, raw_text, is_header=True
        )
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    for row, row_index in table.get("bodyRows", []):
        row, num_row_cols = _ocr_tables_standardize_row(row, raw_text, row_index)
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    return Table(rows=rows, num_rows=len(rows), num_cols=ocr_num_cols)


def _ocr_tables_standardize_row(
    row, raw_text, is_header=False
) -> Tuple[Row, int]:
    cells: Sequence[Cell] = []
    for cell in row.get("cells", []):
        std_cell = _ocr_tables_standardize_cell(cell, raw_text, is_header)
    ocr_row = Row(cells=cells)
    return ocr_row, len(cells)


def _ocr_tables_standardize_cell(cell, raw_text, is_header) -> Cell:
    vertices = cell["layout"]["boundingPoly"]["normalizedVertices"]
    text = ""
    for segment in cell["layout"]["textAnchor"].get("textSegments", []):
        start_index = int(segment.get("startIndex", 0))
        end_index = int(segment.get("endIndex", 0))
        text += raw_text[start_index:end_index]

    return Cell(
        text=text,
        col_index=0,
        row_index=0,
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
