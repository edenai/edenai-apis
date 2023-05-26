from enum import Enum
from typing import Dict, Optional, Sequence, Union
from pydantic import BaseModel, Field, validator

from edenai_apis.features.text.anonymization.category import (
    CategoryType,
)
from edenai_apis.features.text.anonymization.subcategory import (
    FinancialInformationSubCategoryType,
    PersonalInformationSubCategoryType,
    IdentificationNumbersSubCategoryType,
    MiscellaneousSubCategoryType,
    OrganizationSubCategoryType,
    DateAndTimeSubCategoryType,
    LocationInformationSubCategoryType,
    OtherSubCategoryType,
)

SubCategoryType = Union[
    FinancialInformationSubCategoryType,
    PersonalInformationSubCategoryType,
    IdentificationNumbersSubCategoryType,
    MiscellaneousSubCategoryType,
    OrganizationSubCategoryType,
    DateAndTimeSubCategoryType,
    LocationInformationSubCategoryType,
    OtherSubCategoryType,
]


class AnonymizationEntity(BaseModel):
    """This model represents an entity extracted from the text.

    Attributes:
        offset (int): The offset of the entity in the text.
        length (int): The lenght of the entity in the text.
        category (CategoryType): The category of the entity.
        subcategory (SubCategoryType): The subcategory of the entity.
        original_label (str): The original label of the entity.
        content (str): The content of the entity.
    """

    offset: int = Field(..., ge=0)
    length: int = Field(..., gt=0)
    category: CategoryType = Field(...)
    subcategory: SubCategoryType = Field(...)
    original_label: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    confidence_score: Optional[float] = Field(..., ge=0, le=1)

    class Config:
        use_enum_values = True

    @validator("content", pre=True)
    @classmethod
    def content_must_be_str(cls, v):
        if not isinstance(v, str):
            raise TypeError("entity must be a string")
        return v

    @validator("original_label", pre=True)
    @classmethod
    def original_label_must_be_str(cls, v):
        if not isinstance(v, str):
            raise TypeError("original_label must be a string")
        return v

    @validator("content")
    @classmethod
    def content_length_must_be_equal_to_length(cls, v, values):
        if len(v) != values.get("length", 0):
            raise ValueError("content length must be equal to length")
        return v

    @validator("confidence_score")
    @classmethod
    def round_confidence_score(cls, v):
        if v is not None:
            return round(v, 3)
        return v


class AnonymizationDataClass(BaseModel):
    """This model represents the response from the API.

    Attributes:
        result (str): The anonymized text.
        entities (List[AnonymizationEntity]): The entities extracted from the text.
    """

    result: str
    entities: Sequence[AnonymizationEntity] = Field(default_factory=list)

    @validator("result", pre=True)
    def result_must_be_str(cls, v):
        if not isinstance(v, str):
            raise TypeError("result must be a string")
        return v

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["result"]
