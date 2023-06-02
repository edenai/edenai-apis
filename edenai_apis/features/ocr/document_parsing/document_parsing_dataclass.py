from typing import Any, Sequence, Optional
from pydantic import BaseModel, Field, validator
from edenai_apis.utils.bounding_box import BoundingBox


class ItemDocumentParsing(BaseModel):
    key: str
    value: Any
    bounding_box: BoundingBox
    confidence_score: Optional[float] = Field(..., ge=0, le=1)

    @validator("confidence_score")
    @classmethod
    def round_confidence_score(cls, v) -> Optional[float]:
        if v is not None:
            v = round(v, 2)
        return v


class DocumentParsingDataClass(BaseModel):
    fields: Sequence[ItemDocumentParsing] = Field(default_factory=list)
