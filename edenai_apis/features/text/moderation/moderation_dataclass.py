from typing import Optional, Sequence
from enum import Enum
from pydantic import BaseModel, Field, StrictStr, validator

class TextModerationCategoriesMicrosoftEnum(Enum):
    Category1 = "sexually explicit"
    Category2 = "sexually suggestive"
    Category3 = "offensive"

class TextModerationItem(BaseModel):
    label: StrictStr
    likelihood : int

    @staticmethod
    def moderation_processing(confidence):
        if confidence < 0.2:
            return 1
        elif confidence < 0.4:
            return 2
        elif confidence < 0.6:
            return 3
        elif confidence < 0.8:
            return 4
        elif confidence > 0.8:
            return 5
        else:
            return 0

class ModerationDataClass(BaseModel):
    nsfw_likelihood : int
    items: Sequence[TextModerationItem] = Field(default_factory=list)

    @validator('nsfw_likelihood')
    def check_min_max(cls, v):
        if not 0 <= v <= 5:
            raise ValueError("Likelihood walue should be between 0 and 5")
        return v

    @staticmethod
    def calculate_nsfw_likelihood(items: Sequence[TextModerationItem]):
        if len(items) == 0:
            return 0
        return max([item.likelihood for item in items])

