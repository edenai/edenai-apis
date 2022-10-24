from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class InfosSearchDataClass(BaseModel):
    object: StrictStr
    document: int
    score: float


class SearchDataClass(BaseModel):
    items: Sequence[InfosSearchDataClass] = Field(default_factory=list)
