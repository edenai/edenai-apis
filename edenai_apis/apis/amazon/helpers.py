import json
import urllib
from typing import Dict, Sequence, TypeVar

import requests
from edenai_apis.features.ocr import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from trp import Document


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
    standarized_response = OcrTablesAsyncDataClass(pages=pages, num_pages=num_pages)
    return standarized_response


T = TypeVar("T")

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
