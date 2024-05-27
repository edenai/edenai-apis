from pydantic import BaseModel, field_validator

class AiImageDetectionDataClass(BaseModel):
    score: float
    human_probability: float
    ai_probability: float
    version: str
    mime_type: str
    ai_watermark_detected: bool

    @staticmethod
    def set_label_based_on_score(ai_score: float):
        if ai_score > 0.5:
            return "ai-generated"
        else:
            return "original"
        
    @field_validator("ai_probability")
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