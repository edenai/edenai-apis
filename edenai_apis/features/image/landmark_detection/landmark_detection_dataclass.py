from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class LandmarkVertice(BaseModel):
    x: int
    y: int


class LandmarkLatLng(BaseModel):
    latitude: float
    longitude: float


class LandmarkLocation(BaseModel):
    lat_lng: LandmarkLatLng


class LandmarkItem(BaseModel):
    description: StrictStr
    confidence: float
    bounding_box: Sequence[LandmarkVertice] = Field(default_factory=list)
    locations: Sequence[LandmarkLocation] = Field(default_factory=list)


class LandmarkDetectionDataClass(BaseModel):
    items: Sequence[LandmarkItem] = Field(default_factory=list)
