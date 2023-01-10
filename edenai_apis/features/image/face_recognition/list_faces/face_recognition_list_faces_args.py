import os
from typing import Dict
import uuid

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")
def face_recognition_list_faces_arguments() -> Dict:
    return {"collection_id": "test_f28ca5a9-48f4-4267-a8a8-1007c0f41f6f"}
