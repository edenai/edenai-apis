from typing import Sequence
from typing import Tuple

from google.oauth2.utils import enum
import google.auth
import google
import googleapiclient.discovery

from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    Row,
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


def ocr_tables_async_response_add_rows(
    row, raw_text, is_header=False
) -> Tuple[Row, int]:
    num_cols = 0
    ocr_row: Row = Row()
    if "cells" in row.keys():
        cells: Sequence[Cell] = []
        for cell in row["cells"]:
            num_cols += 1
            vertices = cell["layout"]["boundingPoly"]["normalizedVertices"]
            text = ""
            if "textSegments" in cell["layout"]["textAnchor"].keys():
                for segment in cell["layout"]["textAnchor"]["textSegments"]:
                    text = "" + (
                        raw_text[int(segment["startIndex"]) : int(segment["endIndex"])]
                        if "startIndex" in segment.keys()
                        else ""
                    )
            ocr_cell = Cell(
                text=text,
                row_span=cell["rowSpan"],
                col_span=cell["colSpan"],
                confidence=cell["layout"]["confidence"],
                bounding_box=BoundixBoxOCRTable(
                    left=float(vertices[0].get("x", 0)),
                    top=float(vertices[0].get("y", 0)),
                    width=float(vertices[2].get("x", 0) - vertices[0].get("x", 0)),
                    height=float(vertices[2].get("y", 0) - vertices[0].get("y", 0)),
                ),
            )

            cells.append(ocr_cell)
        ocr_row = Row(cells=cells, is_header=is_header)
    return ocr_row, num_cols

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
