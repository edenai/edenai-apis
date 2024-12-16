from typing import Literal
from pydantic import BaseModel, Field


class DeepfakeDetectionDataClass(BaseModel):
    deepfake_score: float = Field(ge=0, le=1)
    prediction: Literal["deepfake", "original"]

    @staticmethod
    def set_label_based_on_score(deepfake_score: float):
        if deepfake_score > 0.5:
            return "deepfake"
        else:
            return "original"
