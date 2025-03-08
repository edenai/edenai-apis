import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(feature_path, "data")


def search_delete_image_arguments(provider_name: str) -> Dict:

    if provider_name == "sentisight":
        project_id = "42874"
    elif provider_name == "nyckel":
        project_id = "yiilyy1cm0sxiw7n"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )

    return {"image_name": "test.jpg", "project_id": project_id}
