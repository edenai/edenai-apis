from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class InfosLanguageDetectionDataClass(BaseModel):
    language: StrictStr
    confidence: float


class LanguageDetectionDataClass(BaseModel):
    items: Sequence[InfosLanguageDetectionDataClass] = Field(default_factory=list)
