from typing import Sequence

from pydantic import BaseModel, Field, StrictStr, validator


class ExplicitItem(BaseModel):
    label: StrictStr
    likelihood: int

class ExplicitContentDataClass(BaseModel):
    nsfw_likelihood: int
    items: Sequence[ExplicitItem] = Field(default_factory=list)

    # TODO reuse validator
    @validator('nsfw_likelihood')
    def check_min_max(cls, v):
        if not 0 <= v <= 5:
            raise ValueError("Value should be between 0 and 5")
        return v

    @staticmethod
    def calculate_nsfw_likelihood(items: Sequence[ExplicitItem]):
        if len(items) == 0:
            return 0
        safe_labels = ("safe", "sfw")
        return max([item.likelihood for item in items if item.label not in safe_labels])
