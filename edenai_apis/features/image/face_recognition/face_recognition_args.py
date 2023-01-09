from io import BufferedReader
import os
from typing import Dict, List
import uuid

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def get_data_files() -> List[BufferedReader]:
    return [open(os.path.join(data_path, f"face_recognition_{num+1}.jpg"), "rb") for num in range(3)]

def face_recognition_arguments() -> Dict:
    return {"file": open(f"{data_path}/face_recognition_1.jpg", "rb"), "collection_id": uuid.uuid4()}
