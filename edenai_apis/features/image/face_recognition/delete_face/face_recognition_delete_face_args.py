from typing import Dict

from ..common_args import COLLECTION_ID


def face_recognition_delete_face_arguments(provider_name: str) -> Dict:
    return {
        "collection_id": COLLECTION_ID,
        "face_id": "211dbf3e-c258-4e58-8a97-0a886a22900d",
    }
