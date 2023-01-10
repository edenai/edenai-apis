from pydantic import BaseModel

class FaceRecognitionDeleteFaceDataClass(BaseModel):
    deleted: bool
