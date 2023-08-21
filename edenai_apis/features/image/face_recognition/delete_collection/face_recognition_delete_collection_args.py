from typing import Dict

from ..common_args import COLLECTION_ID


def face_recognition_delete_collection_arguments(provider_name: str) -> Dict:
    return {"collection_id": COLLECTION_ID}
