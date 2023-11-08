from typing import List

from pydantic import BaseModel


class FaceRecognitionAddFaceDataClass(BaseModel):
    face_ids: List[str]
