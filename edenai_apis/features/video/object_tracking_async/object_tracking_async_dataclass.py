from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class VideoObjectBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class ObjectFrame(BaseModel):
    timestamp: float
    bounding_box: VideoObjectBoundingBox


class ObjectTrack(BaseModel):
    description: StrictStr
    confidence: float
    frames: Sequence[ObjectFrame] = Field(default_factory=list)


class ObjectTrackingAsyncDataClass(BaseModel):
    objects: Sequence[ObjectTrack] = Field(default_factory=list)
