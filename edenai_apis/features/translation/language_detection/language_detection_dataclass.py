from typing import Sequence, Optional

from pydantic import BaseModel, Field, StrictStr


class InfosLanguageDetectionDataClass(BaseModel):
    language: StrictStr
    confidence: Optional[float]


class LanguageDetectionDataClass(BaseModel):
    items: Sequence[InfosLanguageDetectionDataClass] = Field(default_factory=list)
