from typing import List
from pydantic import BaseModel, Field

class FaceRecognitionRecognizedFaceDataClass(BaseModel):
    confidence: float
    face_id: str

class FaceRecognitionRecognizeDataClass(BaseModel):
   items: List[FaceRecognitionRecognizedFaceDataClass] = Field(default_factory=list)
