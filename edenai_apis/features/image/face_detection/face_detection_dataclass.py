from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class FaceLandmarks(BaseModel):
    left_eye: Sequence[float] = Field(default_factory=list)
    left_eye_top: Sequence[float] = Field(default_factory=list)
    left_eye_right: Sequence[float] = Field(default_factory=list)
    left_eye_bottom: Sequence[float] = Field(default_factory=list)
    left_eye_left: Sequence[float] = Field(default_factory=list)
    right_eye: Sequence[float] = Field(default_factory=list)
    right_eye_top: Sequence[float] = Field(default_factory=list)
    right_eye_right: Sequence[float] = Field(default_factory=list)
    right_eye_bottom: Sequence[float] = Field(default_factory=list)
    right_eye_left: Sequence[float] = Field(default_factory=list)
    left_eyebrow_left: Sequence[float] = Field(default_factory=list)
    left_eyebrow_right: Sequence[float] = Field(default_factory=list)
    left_eyebrow_top: Sequence[float] = Field(default_factory=list)
    right_eyebrow_left: Sequence[float] = Field(default_factory=list)
    right_eyebrow_right: Sequence[float] = Field(default_factory=list)
    left_pupil: Sequence[float] = Field(default_factory=list)
    right_pupil: Sequence[float] = Field(default_factory=list)
    nose_tip: Sequence[float] = Field(default_factory=list)
    nose_bottom_right: Sequence[float] = Field(default_factory=list)
    nose_bottom_left: Sequence[float] = Field(default_factory=list)
    mouth_left: Sequence[float] = Field(default_factory=list)
    mouth_right: Sequence[float] = Field(default_factory=list)
    right_eyebrow_top: Sequence[float] = Field(default_factory=list)
    midpoint_between_eyes: Sequence[float] = Field(default_factory=list)
    nose_bottom_center: Sequence[float] = Field(default_factory=list)
    nose_left_alar_out_tip: Sequence[float] = Field(default_factory=list)
    nose_left_alar_top: Sequence[float] = Field(default_factory=list)
    nose_right_alar_out_tip: Sequence[float] = Field(default_factory=list)
    nose_right_alar_top: Sequence[float] = Field(default_factory=list)
    nose_root_left: Sequence[float] = Field(default_factory=list)
    nose_root_right: Sequence[float] = Field(default_factory=list)
    upper_lip: Sequence[float] = Field(default_factory=list)
    under_lip: Sequence[float] = Field(default_factory=list)
    under_lip_bottom: Sequence[float] = Field(default_factory=list)
    under_lip_top: Sequence[float] = Field(default_factory=list)
    upper_lip_bottom: Sequence[float] = Field(default_factory=list)
    upper_lip_top: Sequence[float] = Field(default_factory=list)
    mouth_center: Sequence[float] = Field(default_factory=list)
    mouth_top: Sequence[float] = Field(default_factory=list)
    mouth_bottom: Sequence[float] = Field(default_factory=list)
    left_ear_tragion: Sequence[float] = Field(default_factory=list)
    right_ear_tragion: Sequence[float] = Field(default_factory=list)
    forehead_glabella: Sequence[float] = Field(default_factory=list)
    chin_gnathion: Sequence[float] = Field(default_factory=list)
    chin_left_gonion: Sequence[float] = Field(default_factory=list)
    chin_right_gonion: Sequence[float] = Field(default_factory=list)
    upper_jawline_left: Sequence[float] = Field(default_factory=list)
    mid_jawline_left: Sequence[float] = Field(default_factory=list)
    mid_jawline_right: Sequence[float] = Field(default_factory=list)
    upper_jawline_right: Sequence[float] = Field(default_factory=list)
    left_cheek_center: Sequence[float] = Field(default_factory=list)
    right_cheek_center: Sequence[float] = Field(default_factory=list)
    # unknown: Sequence[float] = None


class FaceEmotions(BaseModel):
    joy: Optional[int]  # all
    sorrow: Optional[int]  # all
    anger: Optional[int]  # all
    surprise: Optional[int]  # all
    disgust: Optional[int]  # amazon, microsoft
    fear: Optional[int]  # amazon, microsoft
    confusion: Optional[int]  # amazon
    calm: Optional[int]  # amazon
    unknown: Optional[int]  # amazon
    neutral: Optional[int]  # microsoft
    contempt: Optional[int]  # microsoft


class FaceHairColor(BaseModel):
    color: StrictStr
    confidence: float


class FaceHair(BaseModel):
    hair_color: Sequence[FaceHairColor] = Field(default_factory=list)  # microsoft
    bald: Optional[float]  # microsoft
    invisible: bool = None  # microsoft


class FaceFacialHair(BaseModel):
    moustache: Optional[float]  # microsoft, amazon
    beard: Optional[float]  # microsoft, amazon
    sideburns: Optional[float]  # microsoft


class FaceBoundingBox(BaseModel):
    x_min: Optional[float]
    x_max: Optional[float]
    y_min: Optional[float]
    y_max: Optional[float]


class FacePoses(BaseModel):
    pitch: Optional[float]  # all
    roll: Optional[float]  # all
    yaw: Optional[float]  # all


class FaceQuality(BaseModel):
    noise: Optional[float]  # microsoft
    exposure: Optional[float]  # microsoft
    blur: Optional[float]  # microsoft
    brightness: Optional[float]  # amazon
    sharpness: Optional[float]  # amazon


class FaceMakeup(BaseModel):
    eye_make: bool = None  # microsoft
    lip_make: bool = None  # microsoft


class FaceFeatures(BaseModel):
    eyes_open: Optional[float]  # amazon
    smile: Optional[float]  # amazon
    mouth_open: Optional[float]  # amazon


class FaceAccessories(BaseModel):
    sunglasses: Optional[float]  # microsoft, amazon
    reading_glasses: Optional[float]  # microsoft
    swimming_goggles: Optional[float]  # microsoft
    face_mask: Optional[float]  # microsoft
    eyeglasses: Optional[float]  # amazon
    headwear: Optional[float]  # google, microsoft


class FaceOcclusions(BaseModel):
    eye_occluded: bool = None  # microsoft
    forehead_occluded: bool = None  # microsoft
    mouth_occluded: bool = None  # microsoft


class FaceItem(BaseModel):
    confidence: float
    landmarks: FaceLandmarks = FaceLandmarks()
    emotions: FaceEmotions = FaceEmotions()
    poses: FacePoses = FacePoses()
    age: Optional[float]
    gender: Optional[StrictStr]
    bounding_box: FaceBoundingBox = FaceBoundingBox()
    hair: FaceHair = FaceHair()
    facial_hair: FaceFacialHair = FaceFacialHair()
    quality: FaceQuality = FaceQuality()
    makeup: FaceMakeup = FaceMakeup()
    accessories: FaceAccessories = FaceAccessories()
    occlusions: FaceOcclusions = FaceOcclusions()
    features: FaceFeatures = FaceFeatures()


class FaceDetectionDataClass(BaseModel):
    items: Sequence[FaceItem] = Field(default_factory=list)
