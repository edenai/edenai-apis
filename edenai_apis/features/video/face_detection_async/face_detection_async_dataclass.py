from typing import Optional, Sequence

from pydantic import BaseModel, Field


class VideoBoundingBox(BaseModel):
    top: Optional[float]
    left: Optional[float]
    height: Optional[float]
    width: Optional[float]


class VideoFacePoses(BaseModel):
    pitch: Optional[float]
    roll: Optional[float]
    yawn: Optional[float]


class FaceAttributes(BaseModel):
    headwear: Optional[float]
    frontal_gaze: Optional[float]
    eyes_visible: Optional[float]
    glasses: Optional[float]
    mouth_open: Optional[float]
    smiling: Optional[float]
    brightness: Optional[float]
    sharpness: Optional[float]
    pose: VideoFacePoses


class LandmarksVideo(BaseModel):
    eye_left: Sequence[float] = Field(default_factory=list)
    eye_right: Sequence[float] = Field(default_factory=list)
    nose: Sequence[float] = Field(default_factory=list)
    mouth_left: Sequence[float] = Field(default_factory=list)
    mouth_right: Sequence[float] = Field(default_factory=list)


class VideoFace(BaseModel):
    offset: Optional[float]
    bounding_box: VideoBoundingBox
    attributes: FaceAttributes
    landmarks: LandmarksVideo


class FaceDetectionAsyncDataClass(BaseModel):
    faces: Sequence[VideoFace] = Field(default_factory=list)
