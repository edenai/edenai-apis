import json
import os
from typing import Sequence, Optional

from pydantic import BaseModel, Field, StrictStr, validator

class InfosLanguageDetectionDataClass(BaseModel):
    language: StrictStr
    display_name: StrictStr
    confidence: Optional[float]

    @validator('confidence')
    @classmethod
    def normalize_confidence(cls, value):
        return round(value, 2)


class LanguageDetectionDataClass(BaseModel):
    items: Sequence[InfosLanguageDetectionDataClass] = Field(default_factory=list)
