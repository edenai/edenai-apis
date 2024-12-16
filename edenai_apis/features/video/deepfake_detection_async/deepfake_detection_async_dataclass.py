from typing import Literal, Sequence
from pydantic import BaseModel, Field


class DetailPerFrame(BaseModel):
    position: float
    score: float = Field(ge=0, le=1)
    prediction: Literal["deepfake", "original"]


class DeepfakeDetectionAsyncDataClass(BaseModel):
    average_score: float = Field(ge=0, le=1)
    prediction: Literal["deepfake", "original"]
    details_per_frame: Sequence[DetailPerFrame] = Field(default_factory=list)

    @staticmethod
    def set_label_based_on_score(deepfake_score: float):
        if deepfake_score > 0.5:
            return "deepfake"
        else:
            return "original"
