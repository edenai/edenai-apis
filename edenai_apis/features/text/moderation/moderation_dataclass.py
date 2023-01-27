from typing import Sequence
from enum import Enum
from pydantic import BaseModel, Field, StrictStr, validator


class TextModerationCategoriesMicrosoftEnum(Enum):
    Category1 = "sexually explicit"
    Category2 = "sexually suggestive"
    Category3 = "offensive"

class TextModerationItem(BaseModel):
    label: StrictStr
    likelihood : int

class ModerationDataClass(BaseModel):
    nsfw_likelihood : int
    items: Sequence[TextModerationItem] = Field(default_factory=list)

    @validator('nsfw_likelihood')
    @classmethod
    def check_min_max(cls, value):
        if not 0 <= value <= 5:
            raise ValueError("Likelihood walue should be between 0 and 5")
        return value

    @staticmethod
    def calculate_nsfw_likelihood(items: Sequence[TextModerationItem]):
        if len(items) == 0:
            return 0
        return max([item.likelihood for item in items])
