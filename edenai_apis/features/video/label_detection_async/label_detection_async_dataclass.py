from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class VideoLabelBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class VideoLabelTimeStamp(BaseModel):
    start: Optional[float]
    end: Optional[float]


class VideoLabel(BaseModel):
    name: StrictStr
    confidence: float
    timestamp: Sequence[VideoLabelTimeStamp] = Field(default_factory=list)
    category: Sequence[str] = Field(default_factory=list)
    bounding_box: Sequence[VideoLabelBoundingBox] = Field(default_factory=list)


class LabelDetectionAsyncDataClass(BaseModel):
    labels: Sequence[VideoLabel] = Field(default_factory=list)
