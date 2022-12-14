from typing import Sequence

from pydantic import BaseModel, Field, StrictStr, validator


class ExtractedTopic(BaseModel):
    category: StrictStr
    confidence: float
    
    @validator('category', pre=True)
    def valid_category(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"Category must be a string, not {type(value)}")
        value = value.title()
        return value

    @validator('confidence', pre=True)
    def valid_confidence(cls, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f"Confidence must be a float, not {type(value)}")
        if value < 0 or value > 1:
            raise ValueError(f"{value} is not allowed. Confidence must be between 0 and 1")
        return round(value, 2)


class TopicExtractionDataClass(BaseModel):
    categories: Sequence[ExtractedTopic] = Field(default_factory=list)