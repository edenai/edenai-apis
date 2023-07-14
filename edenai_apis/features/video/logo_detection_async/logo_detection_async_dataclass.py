from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class VideoLogoBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class VideoLogo(BaseModel):
    timestamp: float
    bounding_box: VideoLogoBoundingBox
    confidence: Optional[float]


class LogoTrack(BaseModel):
    description: StrictStr
    tracking: Sequence[VideoLogo] = Field(default_factory=list)


class LogoDetectionAsyncDataClass(BaseModel):
    logos: Sequence[LogoTrack] = Field(default_factory=list)
