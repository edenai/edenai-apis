from pydantic import BaseModel, StrictStr, Field
from typing import Sequence, Optional


class AnonymizationBoundingBox(BaseModel):
    x_min: Optional[float]
    x_max: Optional[float]
    y_min: Optional[float]
    y_max: Optional[float]


class AnonymizationItem(BaseModel):
    kind: StrictStr
    confidence: float
    bounding_boxes: AnonymizationBoundingBox


class AnonymizationDataClass(BaseModel):
    image: StrictStr
    image_resource_url: StrictStr
    items: Sequence[AnonymizationItem] = Field(default_factory=list)
