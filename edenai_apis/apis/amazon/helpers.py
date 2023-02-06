import json
import urllib
from io import BufferedReader
from time import time
from typing import Dict, List, Optional, TypeVar, Sequence, Union
from pathlib import Path
import requests
from edenai_apis.features.ocr.custom_document_parsing_async.custom_document_parsing_async_dataclass import CustomDocumentParsingAsyncBoundingBox, CustomDocumentParsingAsyncDataClass, CustomDocumentParsingAsyncItem
from edenai_apis.features.text.custom_named_entity_recognition.custom_named_entity_recognition_dataclass import CustomNamedEntityRecognitionDataClass
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
from edenai_apis.utils.exception import ProviderException

from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)

from .config import clients, storage_clients


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
        if response_status != 200 or len(
            webhook_response.json()["data"]
        ) == 0:
            print("status", response_status)
            print("webhook_response.text", webhook_response.text)
            return None, response_status
        return json.loads(webhook_response.json()["data"][0]["content"]), response_status
    except Exception:
        return None, response_status


def amazon_ocr_tables_parser(original_result) -> OcrTablesAsyncDataClass:
    document = Document(original_result)
    pages: Sequence[Page] = []
    num_pages = 0
    for page in document.pages:
        num_pages += 1
        tables: Sequence[Table] = []
        for table in page.tables:
            ocr_num_rows = 0
            rows: Sequence[Row] = []
            ocr_num_cols = 0
            for row in table.rows:
                num_col = 0
                ocr_num_rows += 1
                is_header = False
                cells: Sequence[Cell] = []
                for cell in row.cells:
                    num_col += 1
                    ocr_cell = Cell(
                        text=cell.text,
                        row_span=cell.rowSpan,
                        col_span=cell.columnSpan,
                        bounding_box=BoundixBoxOCRTable(
                            left=cell.geometry.boundingBox.left,
                            top=cell.geometry.boundingBox.top,
                            width=cell.geometry.boundingBox.width,
                            height=cell.geometry.boundingBox.height,
                        ),
                    )
                    if "COLUMN_HEADER" in cell.entityTypes:
                        is_header = True
                    cells.append(ocr_cell)
                ocr_row = Row(cells=cells, is_header=is_header)
                rows.append(ocr_row)
                ocr_num_cols = max(num_col, ocr_num_cols)
            ocr_table = Table(rows=rows, num_cols=ocr_num_cols, num_rows=ocr_num_rows)
            tables.append(ocr_table)
        ocr_page = Page(tables=tables)
        pages.append(ocr_page)
    standardized_response = OcrTablesAsyncDataClass(pages=pages, num_pages=num_pages)
    return standardized_response


T = TypeVar("T")

# Video analysis async
def _upload_video_file_to_amazon_server(file: BufferedReader, file_name: str, api_settings : Dict):
    """
    :param video:       String that contains the video file path
    :return:            String that contains the filename on the server
    """
    # Store file in an Amazon server
    file_extension = file.name.split(".")[-1]
    filename = str(int(time())) + file_name.stem + "_video_." + file_extension
    storage_clients(api_settings)["video"].meta.client.upload_fileobj(file, api_settings['bucket_video'], filename)

    return filename


def amazon_launch_video_job(file: BufferedReader, feature: str):
    api_settings = load_provider(ProviderDataEnum.KEY, "amazon")
    # Upload video to amazon server
    filename = _upload_video_file_to_amazon_server(file, Path(file.name), api_settings)

    # Get response
    role = api_settings['role']
    topic = api_settings['topic_video']
    bucket = api_settings['bucket_video']

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
                bounding_box = CustomDocumentParsingAsyncBoundingBox(
                    left=block["Geometry"]["BoundingBox"]["Left"],
                    top=block["Geometry"]["BoundingBox"]["Top"],
                    width=block["Geometry"]["BoundingBox"]["Width"],
                    height=block["Geometry"]["BoundingBox"]["Height"],
                )
                query = query_answer_result(page["Blocks"], block["Id"])
                item = CustomDocumentParsingAsyncItem(
                    confidence=block["Confidence"],
                    value=block["Text"],
                    query = query,
                    page=block.get("Page", index+1),
                    bounding_box=bounding_box,
                )
                items.append(item)
    return CustomDocumentParsingAsyncDataClass(items=items)


def query_answer_result(page :List[dict], identifier: str):
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
            first_relation_id = relationships[0]['Ids'][0]
            if first_relation_id == identifier:
                return query["Query"]["Text"]
    return None
