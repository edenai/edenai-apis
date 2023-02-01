import os
from typing import Dict

from ..common_args import COLLECTION_ID

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")
def face_recognition_list_faces_arguments() -> Dict:
    return {"collection_id": COLLECTION_ID}
