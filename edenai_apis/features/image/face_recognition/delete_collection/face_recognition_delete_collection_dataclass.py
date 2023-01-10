from pydantic import BaseModel

class FaceRecognitionDeleteCollectionDataClass(BaseModel):
    deleted: bool
