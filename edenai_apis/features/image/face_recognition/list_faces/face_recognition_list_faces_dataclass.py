from typing import List

from pydantic import BaseModel


class FaceRecognitionListFacesDataClass(BaseModel):
    face_ids: List[str]
