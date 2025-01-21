from enum import Enum
from typing import Dict

from edenai_apis.features.text.moderation.subcategory import (
    SubCategoryBase,
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


class CategoryType(str, Enum):
    """This enum are used to categorize the explicit content extracted from the text"""

    Toxic = "Toxic"
    Content = "Content"
    Sexual = "Sexual"
    Violence = "Violence"
    DrugAndAlcohol = "DrugAndAlcohol"
    Finance = "Finance"
    HateAndExtremism = "HateAndExtremism"
    Safe = "Safe"
    Other = "Other"

    @classmethod
    def list_available_type(cls):
        return [category for category in cls]

    @classmethod
    def list_choices(cls) -> Dict[str, SubCategoryBase]:
        return {
            cls.Toxic: ToxicSubCategoryType,
            cls.Content: ContentSubCategoryType,
            cls.Sexual: SexualSubCategoryType,
            cls.Violence: ViolenceSubCategoryType,
            cls.DrugAndAlcohol: DrugAndAlcoholSubCategoryType,
            cls.Finance: FinanceSubCategoryType,
            cls.HateAndExtremism: HateAndExtremismSubCategoryType,
            cls.Safe: SafeSubCategoryType,
            cls.Other: OtherSubCategoryType,
        }

    @classmethod
    def choose_category_subcategory(cls, label) -> dict:
        """Choose the category based on the label.

        Args:
            label (str): The label of the entity.

        Returns:
            CategoryType: The category of the entity.

        """
        for category, sub_categories in cls.list_choices().items():
            try:
                sub_category = sub_categories.choose_label(label)
                return {"category": category, "subcategory": sub_category}
            except ValueError:
                continue
        return {
            "category": cls.Other,
            "subcategory": OtherSubCategoryType.Other,
        }
