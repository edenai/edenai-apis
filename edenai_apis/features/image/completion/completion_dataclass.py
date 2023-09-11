from typing import Sequence
from pydantic import BaseModel, Field


class CompletionDataClass(BaseModel):
    completion: Sequence[str] = Field(default_factory=list)
