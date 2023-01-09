import os
from typing import Dict
import uuid

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")
def face_recognition_arguments() -> Dict:
    return {"file": open(f"{data_path}/face_recognition_1.jpg", "rb"), "collection_id": uuid.uuid4()}
