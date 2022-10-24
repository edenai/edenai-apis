from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class ContentNSFW(BaseModel):
    timestamp: float
    confidence: float
    category: StrictStr


class ExplicitContentDetectionAsyncDataClass(BaseModel):
    moderation: Sequence[ContentNSFW] = Field(default_factory=list)
