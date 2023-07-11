from typing import List, Optional
from pydantic import BaseModel, Field, StrictStr


class CustomDocumentParsingAsyncBoundingBox(BaseModel):
    left: Optional[float]
    top: Optional[float]
    width: Optional[float]
    height: Optional[float]


class CustomDocumentParsingAsyncItem(BaseModel):
    confidence: float
    value: StrictStr
    query: StrictStr
    bounding_box: CustomDocumentParsingAsyncBoundingBox
    page: int


class CustomDocumentParsingAsyncDataClass(BaseModel):
    items: List[CustomDocumentParsingAsyncItem] = Field(default_factory=list)
