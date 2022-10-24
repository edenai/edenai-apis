import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(feature_path, "data")


def search_delete_image_arguments() -> Dict:
    return {"image_name": "test.jpg", "project_id": "42874"}
