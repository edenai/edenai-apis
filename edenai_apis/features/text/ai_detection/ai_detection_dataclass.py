from typing import Sequence

from pydantic import BaseModel, Field, field_validator, model_validator


class AiDetectionItem(BaseModel):
    text: str
    prediction: str
    ai_score: float
    ai_score_detail: float

    @model_validator(mode="before")
    def _set_ai_score_detail(cls, values: dict) -> dict:
        ai_score = values.get("ai_score", None)
        if ai_score is None:
            return values
        values["ai_score_detail"] = ai_score
        return values

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

    @staticmethod
    def set_label_based_on_human_score(human_score: float):
        if human_score > 0.5:
            return "original"
        else:
            return "ai-generated"


class AiDetectionDataClass(BaseModel):
    ai_score: float
    items: Sequence[AiDetectionItem] = Field(default_factory=list)

    @field_validator("ai_score")
    def check_min_max(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v
