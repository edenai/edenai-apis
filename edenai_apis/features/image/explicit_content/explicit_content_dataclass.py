from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class ExplicitItem(BaseModel):
    label: StrictStr
    likelihood: int


class ExplicitContentDataClass(BaseModel):
    items: Sequence[ExplicitItem] = Field(default_factory=list)
