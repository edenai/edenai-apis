from enum import Enum
from typing import Dict, List

from edenai_apis.features.text.moderation.pattern import SubCategoryPattern


class SubCategoryBase(str):
    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        raise NotImplementedError

    @classmethod
    def list_available_type(cls):
        return [category for category in cls]

    @classmethod
    def get_choices(cls, subcategory: "SubCategoryBase") -> list:
        try:
            return cls.list_choices()[subcategory]
        except KeyError:
            raise ValueError(
                f"Unknown subcategory {subcategory}. Only {cls.list_choices().keys()} are allowed."
            )

    @classmethod
    def choose_label(cls, label: str) -> "SubCategoryBase":
        normalized_label = label.lower()
        for subcategory in cls.list_choices().keys():
            choices: list = list(
                map(
                    lambda label: normalized_label == label,
                    cls.get_choices(subcategory),
                )
            )
            if sum(choices) > 0:
                return subcategory
        raise ValueError(
            f"Unknown label {label}. Only {cls.list_choices().values()} are allowed."
        )


class ToxicSubCategoryType(SubCategoryBase, Enum):
    Insult = "Insult"
    Obscene = "Obscene"
    Derogatory = "Derogatory"
    Profanity = "Profanity"
    Threat = "Threat"
    Toxic = "Toxic"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Insult: SubCategoryPattern.Toxic.INSULT,
            cls.Obscene: SubCategoryPattern.Toxic.OBSCENE,
            cls.Derogatory: SubCategoryPattern.Toxic.DEROGATORY,
            cls.Profanity: SubCategoryPattern.Toxic.PROFANITY,
            cls.Threat: SubCategoryPattern.Toxic.THREAT,
            cls.Toxic: SubCategoryPattern.Toxic.TOXIC,
        }


class ContentSubCategoryType(SubCategoryBase, Enum):
    MiddleFinger = "MiddleFinger"
    PublicSafety = "PublicSafety"
    Health = "Health"
    Explicit = "Explicit"
    QRCode = "QRCode"
    Medical = "Medical"
    Politics = "Politics"
    Legal = "Legal"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.MiddleFinger: SubCategoryPattern.Content.MIDDLE_FINGER,
            cls.PublicSafety: SubCategoryPattern.Content.PUBLIC_SAFETY,
            cls.Health: SubCategoryPattern.Content.HEALTH,
            cls.Explicit: SubCategoryPattern.Content.EXPLICIT,
            cls.QRCode: SubCategoryPattern.Content.QRCODE,
            cls.Medical: SubCategoryPattern.Content.MEDICAL,
            cls.Politics: SubCategoryPattern.Content.POLITICS,
            cls.Legal: SubCategoryPattern.Content.LEGAL,
        }


class SexualSubCategoryType(SubCategoryBase, Enum):
    SexualActivity = "SexualActivity"
    SexualSituations = "SexualSituations"
    Nudity = "Nudity"
    PartialNudity = "PartialNudity"
    Suggestive = "Suggestive"
    AdultToys = "AdultToys"
    RevealingClothes = "RevealingClothes"
    Sexual = "Sexual"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.SexualActivity: SubCategoryPattern.Sexual.SEXUAL_ACTIVITY,
            cls.SexualSituations: SubCategoryPattern.Sexual.SEXUAL_SITUATIONS,
            cls.Nudity: SubCategoryPattern.Sexual.NUDITY,
            cls.PartialNudity: SubCategoryPattern.Sexual.PARTIAL_NUDITY,
            cls.Suggestive: SubCategoryPattern.Sexual.SUGGESTIVE,
            cls.AdultToys: SubCategoryPattern.Sexual.ADULT_TOYS,
            cls.RevealingClothes: SubCategoryPattern.Sexual.REVEALING_CLOTHES,
            cls.Sexual: SubCategoryPattern.Sexual.SEXUAL,
        }


class ViolenceSubCategoryType(SubCategoryBase, Enum):
    GraphicViolenceOrGore = "GraphicViolenceOrGore"
    PhysicalViolence = "PhysicalViolence"
    WeaponViolence = "WeaponViolence"
    Violence = "Violence"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.GraphicViolenceOrGore: SubCategoryPattern.Violence.GRAPHIC_VIOLENCE_OR_GORE,
            cls.PhysicalViolence: SubCategoryPattern.Violence.PHYSICAL_VIOLENCE,
            cls.WeaponViolence: SubCategoryPattern.Violence.WEAPON_VIOLENCE,
            cls.Violence: SubCategoryPattern.Violence.VIOLENCE,
        }


class DrugAndAlcoholSubCategoryType(SubCategoryBase, Enum):
    DrugProducts = "DrugProducts"
    DrugUse = "DrugUse"
    Tobacco = "Tobacco"
    Smoking = "Smoking"
    Alcohol = "Alcohol"
    Drinking = "Drinking"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.DrugProducts: SubCategoryPattern.DrugAndAlcohol.DRUG_PRODUCTS,
            cls.DrugUse: SubCategoryPattern.DrugAndAlcohol.DRUG_USE,
            cls.Tobacco: SubCategoryPattern.DrugAndAlcohol.TOBACCO,
            cls.Smoking: SubCategoryPattern.DrugAndAlcohol.SMOKING,
            cls.Alcohol: SubCategoryPattern.DrugAndAlcohol.ALCOHOL,
            cls.Drinking: SubCategoryPattern.DrugAndAlcohol.DRINKING,
        }


class FinanceSubCategoryType(SubCategoryBase, Enum):
    Gambling = "Gambling"
    Finance = "Finance"
    MoneyContent = "MoneyContent"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Gambling: SubCategoryPattern.Finance.GAMBLING,
            cls.Finance: SubCategoryPattern.Finance.FINANCE,
            cls.MoneyContent: SubCategoryPattern.Finance.MONEY_CONTENT,
        }


class HateAndExtremismSubCategoryType(SubCategoryBase, Enum):
    Hate = "Hate"
    Harassment = "Harassment"
    Threatening = "Threatening"
    Extremist = "Extremist"
    Racy = "Racy"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Hate: SubCategoryPattern.HateAndExtremism.HATE,
            cls.Harassment: SubCategoryPattern.HateAndExtremism.HARASSMENT,
            cls.Threatening: SubCategoryPattern.HateAndExtremism.THREATENING,
            cls.Extremist: SubCategoryPattern.HateAndExtremism.EXTREMIST,
            cls.Racy: SubCategoryPattern.HateAndExtremism.RACY,
        }


class SafeSubCategoryType(SubCategoryBase, Enum):
    Safe = "Safe"
    NotSafe = "NotSafe"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Safe: SubCategoryPattern.Safe.SAFE,
            cls.NotSafe: SubCategoryPattern.Safe.NOT_SAFE,
        }


class OtherSubCategoryType(SubCategoryBase, Enum):
    Spoof = "Spoof"
    Religion = "Religion"
    Offensive = "Offensive"
    Other = "Other"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Spoof: SubCategoryPattern.Other.SPOOF,
            cls.Religion: SubCategoryPattern.Other.RELIGION,
            cls.Offensive: SubCategoryPattern.Other.OFFENSIVE,
            cls.Other: SubCategoryPattern.Other.OTHER,
        }
