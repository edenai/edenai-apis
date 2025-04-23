import urllib
from typing import Dict, List

import requests

from edenai_apis.features.image.automl_classification.list_data.automl_classification_list_data_dataclass import (
    AutomlClassificationListEntryDataClass,
)


def check_webhook_result(job_id: str, webhook_settings: dict) -> Dict:
    """
     Try to get result on webhook.site with job id

    Args:
        job_id (str): async job id to get result to

    Returns:
        Dict: Result dict
    """
    webhook_token = webhook_settings["webhook_token"]
    api_key = webhook_settings["webhook_api_key"]
    query = f'content:"{job_id}"'
    webhook_get_url = (
        f"https://webhook.site/token/{webhook_token}/requests"
        + f"?sorting=newest&query={urllib.parse.quote_plus(query)}"
    )
    webhook_response = requests.get(url=webhook_get_url, headers={"Api-Key": api_key})
    response_status = webhook_response.status_code
    try:
        return webhook_response.json().get("data"), response_status
    except Exception:
        return None, response_status


def update_label_names(
    label_names: List = None, samples: List = None
) -> List[AutomlClassificationListEntryDataClass]:
    result: AutomlClassificationListEntryDataClass = []
    if not label_names:
        return samples
    labels = {}
    for label in label_names:
        labels[label["id"]] = label["name"]
    for sample in samples:
        data = {**sample}
        data["annotation"]["label_name"] = (
            labels[sample["annotation"]["labelId"]]
            if sample["annotation"]["labelId"] in labels
            else "unknown"
        )
        if sample.get("prediction"):
            data["prediction"]["label_name"] = (
                labels[sample["prediction"]["labelId"]]
                if sample["prediction"]["labelId"] in labels
                else "unknown"
            )
        result.append(AutomlClassificationListEntryDataClass(**data))
    return result
