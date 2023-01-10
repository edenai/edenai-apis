import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_path = os.path.join(feature_path, "data")
def face_recognition_add_face_arguments() -> Dict:
    return {"file": open(f"{data_path}/face_recognition_1.jpg", "rb"), "collection_id": "test_f28ca5a9-48f4-4267-a8a8-1007c0f41f6f"}
