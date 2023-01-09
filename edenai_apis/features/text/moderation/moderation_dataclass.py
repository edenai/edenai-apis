from typing import Optional, Sequence
from enum import Enum
from pydantic import BaseModel, Field, StrictStr

class TextModerationCategoriesEnum(Enum):
    Category1 = "sexually explicit"
    Category2 = "sexually suggestive"
    Category3 = "offensive"

    # SEXUALLY_EXPLECIT = "sexually explicit"
    # SEXUALLY_SUGGESTIVE = "sexually suggestive"
    # OFFENSIVE = "offensive"

class EmailInfoTextModeration(BaseModel):
    detected: StrictStr
    sub_type : StrictStr
    text: StrictStr
    index: int

class IpaInfoTextModeration(BaseModel):
    sub_type : StrictStr
    text: StrictStr
    index: int

class PhoneInfoTextModeration(BaseModel):
    country_code: StrictStr
    text: StrictStr
    index: int

class AdresseInfoTextModeration(BaseModel):
    text: StrictStr
    index: int

class ItemProfanityTextModeration(BaseModel):
    index_term: int
    term : StrictStr

class ProfanityTextModeration(BaseModel):
    items : Sequence[ItemProfanityTextModeration] = Field(default_factory=list)

class PersonalDataTextModeration(BaseModel):
    emails: Sequence[EmailInfoTextModeration] = Field(default_factory=list)
    ipas: Sequence[IpaInfoTextModeration] = Field(default_factory=list)
    phones: Sequence[PhoneInfoTextModeration] = Field(default_factory=list)
    addresses: Sequence[AdresseInfoTextModeration] = Field(default_factory=list)

class ClassificationTextModeration(BaseModel):
    categorie: StrictStr
    score : float

class TextModerationDataClass(BaseModel):
    personal_data : Optional[PersonalDataTextModeration]
    profanity_terms : Optional[ProfanityTextModeration]
    classification : Sequence[ClassificationTextModeration] = Field(default_factory=list)