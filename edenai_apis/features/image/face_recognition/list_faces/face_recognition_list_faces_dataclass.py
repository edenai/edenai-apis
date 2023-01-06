from typing import List
from pydantic import BaseModel

class FaceRecognitionListFacesDataClass(BaseModel):
    collection_id: str
    face_ids: List[str]
