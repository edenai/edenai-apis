from enum import Enum
from typing import Dict, List

from edenai_apis.features.text.anonymization.pattern import SubCategoryPattern


class SubCategoryBase(str):
    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        raise NotImplementedError

    @classmethod
    def list_available_type(cls):
        return [category for category in cls]

    @classmethod
    def get_choices(cls, subcategory: "SubCategoryBase") -> list:
        """Get the list of possible labels for a given subcategory.

        Args:
            subcategory (PersonalInformationSubCategoryType):
            The subcategory. Only available subcategories are (Name, Age, Email, Phone)

        Returns:
            list: The list of possible labels.

        Raises:
            ValueError: If the subcategory is unknown.
        """
        try:
            return cls.list_choices()[subcategory]
        except KeyError:
            raise ValueError(
                f"Unknown subcategory {subcategory}. Only {cls.list_choices().keys()} are allowed."
            )

    @classmethod
    def choose_label(cls, label: str) -> "SubCategoryBase":
        """Choose the subcategory for a given label.

        Args:
            label (str): The label to categorize.

        Returns:
            PersonalInformationSubCategoryType: The subcategory.

        Raises:
            ValueError: If the label is unknown.
        """

        normalized_label = label.lower()
        for subcategory in cls.list_choices().keys():
            choices: list = list(
                map(
                    lambda label: normalized_label in label,
                    cls.get_choices(subcategory),
                )
            )
            if sum(choices) > 0:
                return subcategory
        raise ValueError(
            f"Unknown label {label}. Only {cls.list_choices().values()} are allowed."
        )


class PersonalInformationSubCategoryType(SubCategoryBase, Enum):
    Name = "Name"
    Age = "Age"
    Email = "Email"
    Phone = "Phone"
    PersonType = "PersonType"
    Gender = "Gender"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Name: SubCategoryPattern.PersonnalInformation.NAME,
            cls.Age: SubCategoryPattern.PersonnalInformation.AGE,
            cls.Email: SubCategoryPattern.PersonnalInformation.EMAIL,
            cls.Phone: SubCategoryPattern.PersonnalInformation.PHONE,
            cls.PersonType: SubCategoryPattern.PersonnalInformation.PERSON_TYPE,
            cls.Gender: SubCategoryPattern.PersonnalInformation.GENDER,
        }


class FinancialInformationSubCategoryType(SubCategoryBase, Enum):
    CreditCard = "CreditCard"
    CardExpiry = "CardExpiry"
    BankAccountNumber = "BankAccountNumber"
    BankRoutingNumber = "BankRoutingNumber"
    SwiftCode = "SwiftCode"
    TaxIdentificationNumber = "TaxIdentificationNumber"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.CreditCard: SubCategoryPattern.FinancialInformation.CREDIT_CARD,
            cls.CardExpiry: SubCategoryPattern.FinancialInformation.CARD_EXPIRY,
            cls.BankAccountNumber: SubCategoryPattern.FinancialInformation.BANK_ACCOUNT_NUMBER,
            cls.BankRoutingNumber: SubCategoryPattern.FinancialInformation.BANK_ROUTING_NUMBER,
            cls.SwiftCode: SubCategoryPattern.FinancialInformation.SWIFT_CODE,
            cls.TaxIdentificationNumber: SubCategoryPattern.FinancialInformation.TAX_ID,
        }


class IdentificationNumbersSubCategoryType(SubCategoryBase, Enum):
    SocialSecurityNumber = "SocialSecurityNumber"
    NationalIdentificationNumber = "NationalIdentificationNumber"
    NationalHealthService = "NationalHealthService"
    ResidentRegistrationNumber = "ResidentRegistrationNumber"
    DriverLicenseNumber = "DriverLicenseNumber"
    PassportNumber = "PassportNumber"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.SocialSecurityNumber: SubCategoryPattern.IdentificationNumbers.SOCIAL_SECURITY,
            cls.NationalIdentificationNumber: SubCategoryPattern.IdentificationNumbers.NATIONAL_ID,
            cls.NationalHealthService: SubCategoryPattern.IdentificationNumbers.HEALTH_SERVICE,
            cls.ResidentRegistrationNumber: SubCategoryPattern.IdentificationNumbers.RESIDENT,
            cls.DriverLicenseNumber: SubCategoryPattern.IdentificationNumbers.DRIVER_LICENSE,
            cls.PassportNumber: SubCategoryPattern.IdentificationNumbers.PASSPORT_NUMBER,
        }


class MiscellaneousSubCategoryType(SubCategoryBase, Enum):
    URL = "URL"
    IP = "IP"
    MAC = "MAC"
    VehicleIdentificationNumber = "VehicleIdentificationNumber"
    LicensePlate = "LicensePlate"
    VoterNumber = "VoterNumber"
    AWSKeys = "AWSKeys"
    AzureKeys = "AzureKeys"
    Password = "Password"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.URL: SubCategoryPattern.Miscellaneous.URL,
            cls.IP: SubCategoryPattern.Miscellaneous.IP,
            cls.MAC: SubCategoryPattern.Miscellaneous.MAC,
            cls.VehicleIdentificationNumber: SubCategoryPattern.Miscellaneous.VEHICLE_REGISTRATION,
            cls.LicensePlate: SubCategoryPattern.Miscellaneous.LICENSE_PLATE,
            cls.VoterNumber: SubCategoryPattern.Miscellaneous.VOTER_NUMBER,
            cls.AWSKeys: SubCategoryPattern.Miscellaneous.AWS_KEYS,
            cls.AzureKeys: SubCategoryPattern.Miscellaneous.AZURE_KEYS,
            cls.Password: SubCategoryPattern.Miscellaneous.PASSWORD,
        }


class OrganizationSubCategoryType(SubCategoryBase, Enum):
    CompanyName = "CompanyName"
    CompanyNumber = "CompanyNumber"
    BuisnessNumber = "BuisnessNumber"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.CompanyName: SubCategoryPattern.Organization.COMPANY_NAME,
            cls.CompanyNumber: SubCategoryPattern.Organization.COMPANY_NUMBER,
            cls.BuisnessNumber: SubCategoryPattern.Organization.BUSINESS_NUMBER,
        }


class DateAndTimeSubCategoryType(SubCategoryBase, Enum):
    Date = "Date"
    Time = "Time"
    DateTime = "DateTime"
    Duration = "Duration"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Duration: SubCategoryPattern.DateAndTime.DURATION,
            cls.DateTime: SubCategoryPattern.DateAndTime.DATE_TIME,
            cls.Time: SubCategoryPattern.DateAndTime.TIME,
            cls.Date: SubCategoryPattern.DateAndTime.DATE,
        }


class LocationInformationSubCategoryType(SubCategoryBase, Enum):
    Address = "Address"
    Location = "Location"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Address: SubCategoryPattern.LocationInformation.ADDRESS,
            cls.Location: SubCategoryPattern.LocationInformation.LOCATION,
        }


class OtherSubCategoryType(SubCategoryBase, Enum):
    Other = "Other"
    Anonymized = "Anonymized"
    Nerd = "Nerd"
    Wsd = "Wsd"
    Unknown = "Unknown"

    @classmethod
    def list_choices(cls) -> Dict["SubCategoryBase", List[str]]:
        return {
            cls.Other: SubCategoryPattern.Other.OTHER,
            cls.Anonymized: SubCategoryPattern.Other.ANONYMIZED,
            cls.Nerd: SubCategoryPattern.Other.NERD,
            cls.Wsd: SubCategoryPattern.Other.WSD,
        }
