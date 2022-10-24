import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_path = os.path.join(feature_path, "data")


def search_launch_similarity_arguments() -> Dict:
    return {"file": open(f"{data_path}/objects.png", "rb"), "project_id": "42874"}
