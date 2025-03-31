from typing import Sequence, Union, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    field_serializer,
    FieldSerializationInfo,
)

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
from edenai_apis.utils.combine_enums import combine_enums


SubCategoryType = combine_enums(
    "SubCategoryType",
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


class ExplicitItem(BaseModel):
    label: StrictStr = Field(description="")
    likelihood: int = Field(description="")
    likelihood_score: float = Field(description="")
    category: CategoryType = Field(
        description="The category of the detected content. Possible values include: 'Toxic', 'Content', 'Sexual', 'Violence', 'DrugAndAlcohol', 'Finance', 'HateAndExtremism', 'Safe', 'Other'."
    )
    subcategory: Optional[str] = Field(
        description="The subcategory of content. Possible values:\n\n"
        "Toxic Subcategories:\n"
        "- Insult\n"
        "- Obscene\n"
        "- Derogatory\n"
        "- Profanity\n"
        "- Threat\n"
        "- Toxic\n\n"
        "Content Subcategories:\n"
        "- MiddleFinger\n"
        "- PublicSafety\n"
        "- Health\n"
        "- Explicit\n"
        "- QRCode\n"
        "- Medical\n"
        "- Politics\n"
        "- Legal\n\n"
        "Sexual Subcategories:\n"
        "- SexualActivity\n"
        "- SexualSituations\n"
        "- Nudity\n"
        "- PartialNudity\n"
        "- Suggestive\n"
        "- AdultToys\n"
        "- RevealingClothes\n"
        "- Sexual\n\n"
        "Violence Subcategories:\n"
        "- GraphicViolenceOrGore\n"
        "- PhysicalViolence\n"
        "- WeaponViolence\n"
        "- Violence\n\n"
        "Drug and Alcohol Subcategories:\n"
        "- DrugProducts\n"
        "- DrugUse\n"
        "- Tobacco\n"
        "- Smoking\n"
        "- Alcohol\n"
        "- Drinking\n\n"
        "Finance Subcategories:\n"
        "- Gambling\n"
        "- Finance\n"
        "- MoneyContent\n\n"
        "Hate and Extremism Subcategories:\n"
        "- Hate\n"
        "- Harassment\n"
        "- Threatening\n"
        "- Extremist\n"
        "- Racy\n\n"
        "Safe Subcategories:\n"
        "- Safe\n"
        "- NotSafe\n\n"
        "Other Subcategories:\n"
        "- Spoof\n"
        "- Religion\n"
        "- Offensive\n"
        "- Other"
    )
    model_config = ConfigDict(use_enum_values=True)

    @field_serializer("subcategory", mode="plain", when_used="always")
    def serialize_subcategory(self, value: SubCategoryType, _: FieldSerializationInfo):
        return getattr(value, "value", None)


class ExplicitContentDataClass(BaseModel):
    nsfw_likelihood: int = Field(
        description="An integer representing the likelihood of NSFW content. Higher values indicate a higher likelihood."
    )
    nsfw_likelihood_score: float = Field(
        description="A floating-point score representing the confidence level of the NSFW likelihood assessment. This is typically a value between 0.0 and 1.0."
    )
    items: Sequence[ExplicitItem] = Field(
        default_factory=list,
        description="A list of items identified as potentially explicit. Each item contains details of the explicit content detected.",
    )

    @field_validator("nsfw_likelihood")
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

    @staticmethod
    def calculate_nsfw_likelihood_score(items: Sequence[ExplicitItem]):
        if len(items) == 0:
            return 0
        safe_labels = ("safe", "sfw")
        return max(
            [item.likelihood_score for item in items if item.label not in safe_labels]
        )

    @field_validator("nsfw_likelihood_score")
    def check_min_max_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v
