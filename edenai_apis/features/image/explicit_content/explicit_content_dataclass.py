from typing import Sequence, Union

from pydantic import BaseModel, Field, StrictStr, field_validator, field_serializer, FieldSerializationInfo

from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.explicit_content.subcategory import (
    ToxicSubCategoryType,
    ContentSubCategoryType,
    SexualSubCategoryType,
    ViolenceSubCategoryType,
    DrugAndAlcoholSubCategoryType,
    FinanceSubCategoryType,
    HateAndExtremismSubCategoryType,
    SafeSubCategoryType,
    OtherSubCategoryType,
)
SubCategoryType = Union[
    ToxicSubCategoryType,
    ContentSubCategoryType,
    SexualSubCategoryType,
    ViolenceSubCategoryType,
    DrugAndAlcoholSubCategoryType,
    FinanceSubCategoryType,
    HateAndExtremismSubCategoryType,
    SafeSubCategoryType,
    OtherSubCategoryType,
]

class ExplicitItem(BaseModel):
    label: StrictStr
    likelihood: float
    category: CategoryType
    subcategory: SubCategoryType

    @field_serializer('subcategory', mode="plain", when_used="always")
    def serialize_subcategory(self, value: SubCategoryType, _: FieldSerializationInfo):
        return value.value

class ExplicitContentDataClass(BaseModel):
    nsfw_likelihood: float
    items: Sequence[ExplicitItem] = Field(default_factory=list)

    @field_validator("nsfw_likelihood")
    def check_min_max(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v

    @staticmethod
    def calculate_nsfw_likelihood(items: Sequence[ExplicitItem]):
        if len(items) == 0:
            return 0
        safe_labels = ("safe", "sfw")
        return max([item.likelihood for item in items if item.label not in safe_labels])
