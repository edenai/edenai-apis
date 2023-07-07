from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class UpperCloth(BaseModel):
    value: StrictStr
    confidence: float


class LowerCloth(BaseModel):
    value: StrictStr
    confidence: float


class PersonAttributes(BaseModel):
    upper_cloths: Sequence[UpperCloth] = Field(default_factory=list)
    lower_cloths: Sequence[LowerCloth] = Field(default_factory=list)


class VideoTrackingBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class PersonLandmarks(BaseModel):
    eye_left: Sequence[float] = Field(default_factory=list)
    eye_right: Sequence[float] = Field(default_factory=list)
    nose: Sequence[float] = Field(default_factory=list)
    ear_left: Sequence[float] = Field(default_factory=list)
    ear_right: Sequence[float] = Field(default_factory=list)
    shoulder_left: Sequence[float] = Field(default_factory=list)
    shoulder_right: Sequence[float] = Field(default_factory=list)
    elbow_left: Sequence[float] = Field(default_factory=list)
    elbow_right: Sequence[float] = Field(default_factory=list)
    wrist_left: Sequence[float] = Field(default_factory=list)
    wrist_right: Sequence[float] = Field(default_factory=list)
    hip_left: Sequence[float] = Field(default_factory=list)
    hip_right: Sequence[float] = Field(default_factory=list)
    knee_left: Sequence[float] = Field(default_factory=list)
    knee_right: Sequence[float] = Field(default_factory=list)
    ankle_left: Sequence[float] = Field(default_factory=list)
    ankle_right: Sequence[float] = Field(default_factory=list)
    mouth_left: Sequence[float] = Field(default_factory=list)
    mouth_right: Sequence[float] = Field(default_factory=list)


class VideoPersonPoses(BaseModel):
    pitch: Optional[float]
    roll: Optional[float]
    yaw: Optional[float]

    @staticmethod
    def default() -> "VideoPersonPoses":
        return VideoPersonPoses(
            pitch=None,
            roll=None,
            yaw=None,
        )


class VideoPersonQuality(BaseModel):
    brightness: Optional[float]
    sharpness: Optional[float]

    @staticmethod
    def default() -> "VideoPersonQuality":
        return VideoPersonQuality(
            brightness=None,
            sharpness=None,
        )


class PersonTracking(BaseModel):
    offset: float
    attributes: PersonAttributes = Field(default_factory=PersonAttributes)
    landmarks: PersonLandmarks = Field(default_factory=PersonLandmarks)
    poses: VideoPersonPoses = Field(default_factory=VideoPersonPoses)
    quality: VideoPersonQuality = Field(default_factory=VideoPersonQuality)
    bounding_box: VideoTrackingBoundingBox


class VideoTrackingPerson(BaseModel):
    tracked: Sequence[PersonTracking] = Field(default_factory=list)


class PersonTrackingAsyncDataClass(BaseModel):
    persons: Sequence[VideoTrackingPerson] = Field(default_factory=list)
