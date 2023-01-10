from typing import Optional, Sequence
from enum import Enum
from pydantic import BaseModel, Field, StrictStr

class TextModerationCategoriesMicrosoftEnum(Enum):
    Category1 = "sexually explicit"
    Category2 = "sexually suggestive"
    Category3 = "offensive"

class ClassificationTextModeration(BaseModel):
    categorie: StrictStr
    score : float

class ModerationDataClass(BaseModel):
    classification : Sequence[ClassificationTextModeration] = Field(default_factory=list)