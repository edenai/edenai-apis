from io import BufferedReader
import os
from typing import Dict, List

from ..common_args import COLLECTION_ID

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_path = os.path.join(feature_path, "data")

def get_data_files() -> List[BufferedReader]:
    return [os.path.join(data_path, f"face_recognition_{num+1}.jpg") for num in range(3)]

def face_recognition_recognize_arguments() -> Dict:
    return {"file": f"{data_path}/face_recognition_1.jpg", "collection_id": COLLECTION_ID}
