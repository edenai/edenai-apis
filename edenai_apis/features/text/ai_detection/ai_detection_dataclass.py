from typing import Dict, Sequence
from pydantic import BaseModel, Field, field_validator


class AiDetectionItem(BaseModel):
    text: str
    prediction: str
    ai_score: float

    @staticmethod
    def set_label_based_on_score(ai_score: float):
        if ai_score > 0.5:
            return "ai-generated"
        else:
            return "original"

    @field_validator("ai_score")
    def check_min_max(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v


class AiDetectionDataClass(BaseModel):
    ai_score: float
    items: Sequence[AiDetectionItem] = Field(default_factory=list)

    @field_validator("ai_score")
    def check_min_max(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v
