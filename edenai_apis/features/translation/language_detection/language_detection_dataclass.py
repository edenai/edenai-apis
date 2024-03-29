from typing import Sequence, Optional

from pydantic import BaseModel, Field, StrictStr, field_validator


class InfosLanguageDetectionDataClass(BaseModel):
    language: StrictStr
    display_name: StrictStr
    confidence: Optional[float]

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, value):
        if value is None:
            return value
        return round(value, 2)


class LanguageDetectionDataClass(BaseModel):
    items: Sequence[InfosLanguageDetectionDataClass] = Field(default_factory=list)
