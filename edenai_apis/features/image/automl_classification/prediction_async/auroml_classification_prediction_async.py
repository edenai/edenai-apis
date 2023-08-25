from typing import Sequence
from pydantic import BaseModel, Field


class AutomlClassificationPrediction(BaseModel):
    classes: Sequence[str] = Field(default_factory=list)
    confidences: Sequence[float] = Field(default_factory=list)
