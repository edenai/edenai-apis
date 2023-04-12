from typing import List, Sequence
from typing import Tuple

import enum
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
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import SentimentEnum
from edenai_apis.utils.conversion import convert_pitch_from_percentage_to_semitones


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


def google_ocr_tables_standardize_response(original_response: dict) -> OcrTablesAsyncDataClass:
    """Standardize ocr table with dataclass from given google response"""
    raw_text: str = original_response["text"]
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
        row, num_row_cols = _ocr_tables_standardize_row(
            row, raw_text, is_header=True
        )
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    for row in table.get("bodyRows", []):
        row, num_row_cols = _ocr_tables_standardize_row(row, raw_text)
        ocr_num_cols = max(ocr_num_cols, num_row_cols)
        rows.append(row)
    return Table(rows=rows, num_rows=len(rows), num_cols=ocr_num_cols)


def _ocr_tables_standardize_row(
    row, raw_text, is_header=False
) -> Tuple[Row, int]:
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



def get_formated_speaking_rate(speaking_rate: int):
    if speaking_rate > 100:
        speaking_rate = 100
    if speaking_rate < -100:
        speaking_rate = -100
    if speaking_rate >= 0:
        diff = speaking_rate / 100
        return 1+diff
    diff = -1/2 * speaking_rate / 100
    return 1-diff

def get_formated_speaking_volume(speaking_volume: int):
    if speaking_volume > 100:
        speaking_volume = 100
    if speaking_volume < -100:
        speaking_volume = -100
    return (speaking_volume * 6 / 100)


def generate_tts_params(speaking_rate, speaking_pitch, speaking_volume):
    attribs = {
        "speaking_rate": (speaking_rate, get_formated_speaking_rate(speaking_rate)),
        "pitch": (speaking_pitch, convert_pitch_from_percentage_to_semitones(speaking_pitch)),
        "volume_gain_db": (speaking_volume, get_formated_speaking_volume(speaking_volume))
    }
    params = {}

    for k,v in attribs.items():
        if not v[0]:
            continue
        params[k] = v[1]
    return params


def get_right_audio_support_and_sampling_rate(audio_format: str, list_audio_formats: List):
    extension = audio_format
    if not audio_format:
        audio_format = "mp3"
    if audio_format == "wav":
        audio_format = "wav-linear16"
    if "-" in audio_format:
        extension, audio_format = audio_format.split("-")
    right_audio_format = next(filter(lambda x: audio_format in x.lower(), list_audio_formats), None)
    return extension, right_audio_format
