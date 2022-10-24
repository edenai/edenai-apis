from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class LogoVertice(BaseModel):
    x: float
    y: float


class LogoBoundingPoly(BaseModel):
    vertices: Sequence[LogoVertice]


class LogoItem(BaseModel):
    bounding_poly: Optional[LogoBoundingPoly]
    description: Optional[StrictStr]
    score: Optional[float]


class LogoDetectionDataClass(BaseModel):
    items: Sequence[LogoItem] = Field(default_factory=list)
