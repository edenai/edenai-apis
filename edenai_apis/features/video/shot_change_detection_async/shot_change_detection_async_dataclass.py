from pydantic import BaseModel, Field, StrictStr
from typing import Sequence


class ShotFrame(BaseModel):
    startTimeOffset: float
    endTimeOffset: float


class ShotChangeDetectionAsyncDataClass(BaseModel):
    shotAnnotations: Sequence[ShotFrame] = Field(default_factory=list)
