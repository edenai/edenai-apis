from enum import Enum
from typing import Dict

from edenai_apis.features.text.anonymization.subcategory import (
    FinancialInformationSubCategoryType,
    PersonalInformationSubCategoryType,
    SubCategoryBase,
    IdentificationNumbersSubCategoryType,
    MiscellaneousSubCategoryType,
    OrganizationSubCategoryType,
    DateAndTimeSubCategoryType,
    LocationInformationSubCategoryType,
    OtherSubCategoryType,
)


class CategoryType(str, Enum):
    """This enum are used to categorize the entities extracted from the text."""

    PersonalInformation = "PersonalInformation"
    FinancialInformation = "FinancialInformation"
    IdentificationNumbers = "IdentificationNumbers"
    Miscellaneous = "Miscellaneous"
    OrganizationInformation = "OrganizationInformation"
    DateAndTime = "DateAndTime"
    LocationInformation = "LocationInformation"
    Other = "Other"

    @classmethod
    def list_available_type(cls):
        return [category for category in cls]

    @classmethod
    def list_choices(cls) -> Dict[str, SubCategoryBase]:
        return {
            cls.PersonalInformation: PersonalInformationSubCategoryType,
            cls.FinancialInformation: FinancialInformationSubCategoryType,
            cls.IdentificationNumbers: IdentificationNumbersSubCategoryType,
            cls.Miscellaneous: MiscellaneousSubCategoryType,
            cls.OrganizationInformation: OrganizationSubCategoryType,
            cls.DateAndTime: DateAndTimeSubCategoryType,
            cls.LocationInformation: LocationInformationSubCategoryType,
            cls.Other: OrganizationSubCategoryType,
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
            "subcategory": OtherSubCategoryType.Unknown,
        }
