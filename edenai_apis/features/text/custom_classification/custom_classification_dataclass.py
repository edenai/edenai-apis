from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class ItemCustomClassificationDataClass(BaseModel):
    input: StrictStr
    label: StrictStr
    confidence: float


class CustomClassificationDataClass(BaseModel):
    classifications: Sequence[ItemCustomClassificationDataClass] = Field(
        default_factory=list
    )
