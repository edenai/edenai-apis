from typing import List
from pydantic import BaseModel, Field, StrictStr


class CustomDocumentParsingAsyncBoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float

class CustomDocumentParsingAsyncItem(BaseModel):
    confidence: float
    value: StrictStr
    bounding_box: CustomDocumentParsingAsyncBoundingBox
    page: int

class CustomDocumentParsingAsyncDataClass(BaseModel):
    items: List[CustomDocumentParsingAsyncItem] = Field(default_factory=list)
