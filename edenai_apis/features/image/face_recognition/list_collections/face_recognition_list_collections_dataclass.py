from typing import List

from pydantic import BaseModel, Field


class FaceRecognitionListCollectionsDataClass(BaseModel):
    collections: List[str] = Field(default_factory=list)
