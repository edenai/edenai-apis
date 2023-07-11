from typing import Sequence
from pydantic import BaseModel, StrictStr, Field


class PromptDataClass(BaseModel):
    text: StrictStr


class PromptOptimizationDataClass(BaseModel):
    items: Sequence[PromptDataClass] = Field(default_factory=list)
