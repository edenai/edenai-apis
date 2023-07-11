from typing import Optional, Sequence

from pydantic import BaseModel, Field


class FaceCompareBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class FaceMatch(BaseModel):
    confidence: float
    bounding_box: FaceCompareBoundingBox


class FaceCompareDataClass(BaseModel):
    items: Sequence[FaceMatch] = Field(default_factory=list)
