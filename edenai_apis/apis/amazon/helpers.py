import json
import urllib
from io import BufferedReader
from time import time
from typing import Dict, Tuple, TypeVar, Sequence
from pathlib import Path
import requests
from trp import Document

from edenai_apis.features.ocr import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)

from .config import clients, storage_clients, api_settings


def content_processing(confidence):
    if confidence < 10:
        return 1
    elif confidence < 30:
        return 2
    elif confidence < 60:
        return 3
    elif confidence < 80:
        return 4
    elif 80 < confidence:
        return 5
    else:
        return 0


def check_webhook_result(job_id: str) -> Dict:
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
    try:
        if webhook_response.status_code != 200 or len(
            webhook_response.json()["data"]
        ) == 0:
            print("status", webhook_response.status_code)
            print("webhook_response.text", webhook_response.text)
            return None
        return json.loads(webhook_response.json()["data"][0]["content"])
    except Exception:
        return None




T = TypeVar("T")

# Video analysis async
def _upload_video_file_to_amazon_server(file: BufferedReader, file_name: str):
    """
    :param video:       String that contains the video file path
    :return:            String that contains the filename on the server
    """
    # Store file in an Amazon server
    file_extension = file.name.split(".")[-1]
    filename = str(int(time())) + file_name.stem + "_video_." + file_extension
    storage_clients["video"].meta.client.upload_fileobj(file, api_settings['bucket_video'], filename)

    return filename


def amazon_launch_video_job(file: BufferedReader, feature: str):
    # Upload video to amazon server
    filename = _upload_video_file_to_amazon_server(file, Path(file.name))

    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")

    # Get response
    role = api_settings['role']
    topic = api_settings['topic_video']
    bucket = api_settings['bucket_video']

    features = {
        "LABEL": clients["video"].start_label_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "TEXT": clients["video"].start_text_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "FACE": clients["video"].start_face_detection(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "PERSON": clients["video"].start_person_tracking(
            Video={"S3Object": {"Bucket": bucket, "Name": filename}},
            NotificationChannel={
                "RoleArn": role,
                "SNSTopicArn": topic,
            },
        ),
        "EXPLICIT": clients["video"].start_content_moderation(
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


def amazon_video_response_formatter(
    response: Dict, standarized_response: T, provider_job_id: str
) -> AsyncBaseResponseType[T]:
    if response["JobStatus"] == "SUCCEEDED":
        return AsyncResponseType[T](
            original_response=response,
            standarized_response=standarized_response,
            provider_job_id=provider_job_id,
        )

    elif response["JobStatus"] == "IN_PROGRESS":
        return AsyncPendingResponseType[T](provider_job_id=provider_job_id)
    return AsyncErrorResponseType[T](provider_job_id=provider_job_id)



def amazon_ocr_tables_parser(original_result) -> OcrTablesAsyncDataClass:
    document = Document(original_result)
    std_pages = [_ocr_tables_standarize_page(page) for page in document.pages]
    return OcrTablesAsyncDataClass(pages=std_pages, num_pages=len(std_pages))


def _ocr_tables_standarize_page(page) -> Page:
    std_tables = [_ocr_tables_standarize_table(table) for table in page.tables]
    return Page(tables=std_tables)


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
    is_header = False
    cells: Sequence[Cell] = []
    for cell in row.cells:
        is_header = "COLUMN_HEADER" in cell.entityTypes
        std_cell = _ocr_tables_standarize_cell(cell)
        cells.append(std_cell)

    num_col = len(cells)
    return Row(cells=cells, is_header=is_header), num_col


def _ocr_tables_standarize_cell(cell) -> Cell:
    return Cell(
        text=cell.mergedText,
        row_span=cell.rowSpan,
        col_span=cell.columnSpan,
        confidence=cell.confidence,
        bounding_box=BoundixBoxOCRTable(
            left=cell.geometry.boundingBox.left,
            top=cell.geometry.boundingBox.top,
            width=cell.geometry.boundingBox.width,
            height=cell.geometry.boundingBox.height,
        ),
    )
