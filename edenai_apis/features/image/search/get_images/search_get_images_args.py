import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(feature_path, "data")


def search_get_images_arguments() -> Dict:
    return {"project_id": "42874"}
