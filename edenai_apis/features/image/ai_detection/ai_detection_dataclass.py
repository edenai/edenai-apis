from pydantic import BaseModel, Field


class AiImageDetectionDataClass(BaseModel):
    ai_score: float = Field(ge=0, le=1)
    prediction: str
    ai_watermark_detected: bool

    @staticmethod
    def set_label_based_on_score(ai_score: float):
        if ai_score > 0.5:
            return "ai-generated"
        else:
            return "original"
