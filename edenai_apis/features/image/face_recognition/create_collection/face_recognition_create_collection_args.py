import os
from typing import Dict
import uuid

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")
def face_recognition_create_collection_arguments() -> Dict:
    return {"collection_id": uuid.uuid4()}
