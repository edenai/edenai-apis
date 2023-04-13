from typing import List, Sequence
from typing import Tuple

import enum
import google.auth
import google
import googleapiclient.discovery

from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    Row,
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

#*****************************Speech to text***************************************************
def get_encoding_and_sample_rate(extension: str):
    list_encoding = [("LINEAR16", None), ("MULAW", None), ("AMR", 8000), ("AMR_WB", 16000), 
                     ("OGG_OPUS", 24000), ("SPEEX_WITH_HEADER_BYTE", 16000), ("WEBM_OPUS", 24000)]
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
    right_encoding_sample: Tuple = next(filter(lambda x: extension in x[0].lower(), list_encoding), (None, None))
    return right_encoding_sample


#*****************************Text to speech**************************************************#
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
    
#**************************************************************************************************#