from io import BufferedReader
import os
from typing import Dict, List

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_path = os.path.join(feature_path, "data")

def get_data_files() -> List[BufferedReader]:
    return [open(os.path.join(data_path, f"face_recognition_{num+1}.jpg"), "rb") for num in range(3)]

def face_recognition_recognize_arguments() -> Dict:
    return {"file": open(f"{data_path}/face_recognition_1.jpg", "rb"), "collection_id": "test_f28ca5a9-48f4-4267-a8a8-1007c0f41f6f"}
