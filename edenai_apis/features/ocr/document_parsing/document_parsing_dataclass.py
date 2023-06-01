from typing import Any, Sequence, Optional
from pydantic import BaseModel, Field
from edenai_apis.utils.bounding_box import BoundingBox


class ItemDocumentParsing(BaseModel):
    key: str
    value: Any
    bounding_box: BoundingBox
    confidence_score: Optional[float] = Field(..., ge=0, le=1)


class DocumentParsingDataClass(BaseModel):
    fields: Sequence[ItemDocumentParsing] = Field(default_factory=list)
