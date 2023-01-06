from pydantic import BaseModel

class FaceRecognitionCreateCollectionDataClass(BaseModel):
    collection_id: str
