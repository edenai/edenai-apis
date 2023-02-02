import os
from typing import Dict

from ..common_args import COLLECTION_ID

feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

data_path = os.path.join(feature_path, "data")
def face_recognition_add_face_arguments() -> Dict:
    return {"file": open(f"{data_path}/face_recognition_1.jpg", "rb"), "collection_id": COLLECTION_ID}
