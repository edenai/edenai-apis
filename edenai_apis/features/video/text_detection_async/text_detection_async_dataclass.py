from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class VideoTextBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class VideoTextFrames(BaseModel):
    confidence: float
    timestamp: float
    bounding_box: VideoTextBoundingBox


class VideoText(BaseModel):
    text: StrictStr
    frames: Sequence[VideoTextFrames] = Field(default_factory=list)


class TextDetectionAsyncDataClass(BaseModel):
    texts: Sequence[VideoText] = Field(default_factory=list)
