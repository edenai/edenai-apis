class SubCategoryPattern:
    """This class contains all the patterns for the subcategories of the anonymization category.

    Subclasses:
        * PersonalInformation: This class contains a constant for each subcategory of the personal information category.
        * FinancialInformation: This class contains a constant for each subcategory of the financial information category.
        * IdentificationNumbers: This class contains a constant for each subcategory of the identification numbers category.
        * Miscellaneous: This class contains a constant for each subcategory of the miscellaneous category.
        * OrganizationInformation: This class contains a constant for each subcategory of the organization information category.
        * DateAndTime: This class contains a constant for each subcategory of the date and time category.
        * LocationInformation: This class contains a constant for each subcategory of the location information category.
        * Other: This class contains a constant for each subcategory of the other category.
    """

    class PersonnalInformation:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            NAME (list): List of all the patterns for the name subcategory.
            AGE (list): List of all the patterns for the age subcategory.
            EMAIL (list): List of all the patterns for the email subcategory.
            PHONE (list): List of all the patterns for the phone subcategory.
            PERSON_TYPE (list): List of all the patterns for the person type subcategory.
        """

        NAME = ["name", "username", "person", "human", "name_given", "name_family"]
        AGE = ["age"]
        EMAIL = ["email", "email_address"]
        PHONE = ["phone", "phonenumber", "phone number", "phone_number"]
        PERSON_TYPE = ["persontype"]
        GENDER = ["gender_sexuality"]

    class FinancialInformation:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            CREDIT_CARD (list): List of all the patterns for the credit card subcategory.
            CARD_EXPIRY (list): List of all the patterns for the card expiry subcategory.
            BANK_ACCOUNT_NUMBER (list): List of all the patterns for the bank account number subcategory.
            BANK_ROUTING_NUMBER (list): List of all the patterns for the bank routing number subcategory.
            SWIFT_CODE (list): List of all the patterns for the swift code subcategory.
            TAX_ID (list): List of all the patterns for the tax id subcategory.
        """

        CREDIT_CARD = [
            "credit_debit_cvv",
            "credit_debit_number",
            "credit card",
            "debit card number",
        ]
        CARD_EXPIRY = ["credit_debit_expiry"]
        BANK_ACCOUNT_NUMBER = [
            "bank account number",
            "bank_account_number",
            "international_bank_account_number",
            "account_number",
        ]
        BANK_ROUTING_NUMBER = ["bank_routing", "aba_routing_number", "iban"]
        SWIFT_CODE = ["swift code", "swift_code"]
        TAX_ID = [
            "taxpayer_reference_number",
            "tax identification number",
            "value added tax (vat) number",
            "tax file number",
            "fiscal code",
            "inland revenue number",
        ]

    class IdentificationNumbers:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            SOCIAL_SECURITY (list): List of all the patterns for the social security subcategory.
            NATIONAL_ID (list): List of all the patterns for the national id subcategory.
            HEALTH_SERVICE (list): List of all the patterns for the health service subcategory.
            RESIDENT (list): List of all the patterns for the resident subcategory.
            DRIVING_LICENSE (list): List of all the patterns for the driving license subcategory.
            PASSPORT_NUMBER (list): List of all the patterns for the passport number subcategory.
        """

        SOCIAL_SECURITY = [
            "social_security_number",
            "ssn",
            "social_insurance_number",
            "social insurance number",
            "social security number",
            "social welfare number",
            "insurance_number",
            "insurance number",
            "medical account number",
        ]
        NATIONAL_ID = [
            "national identity",
            "national number",
            "national id card",
            "national identification number",
            "national id",
            "national registration id card",
            "dni",
            "aadhaar",
            "identity card",
            "personal identification number",
            "permanent account number (pan)",
            "personal public service",
            "unified multi-purpose id number",
            "citizen card number",
            "identity number",
            "permament_account_number",
            "nrega",
        ]
        HEALTH_SERVICE = [
            "health_number",
            "health_service_number",
            "health service number",
            "personal health identification number",
            "health insurance number",
            "health number",
        ]
        RESIDENT = [
            "resident identity card",
            '"my number" card number',
            "resident registration number",
            "residence card number",
            "resident certificate",
        ]
        DRIVER_LICENSE = ["driver_id", "driver's license"]
        PASSPORT_NUMBER = ["passport_number", "passport number"]

    class Miscellaneous:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            URL (list): List of all the patterns for the url subcategory.
            IP (list): List of all the patterns for the ip subcategory.
            MAC (list): List of all the patterns for the mac subcategory.
            VEHICLE_REGISTRATION (list): List of all the patterns for the vehicle registration subcategory.
            LICENSE_PLATE (list): List of all the patterns for the license plate subcategory.
            VOTER_NUMBER (list): List of all the patterns for the voter number subcategory.
            AWS_KEYS (list): List of all the patterns for the aws keys subcategory.
            AZURE_KEYS (list): List of all the patterns for the azure keys subcategory.
            PASSWORD (list): List of all the patterns for the password subcategory.
        """

        URL = ["url"]
        IP = ["ip address", "ip_address"]
        MAC = ["mac_address"]
        VEHICLE_REGISTRATION = ["vehicle_identification_number", "vehicle"]
        LICENSE_PLATE = ["license_plate", "license_plate_number"]
        VOTER_NUMBER = ["voter"]
        AWS_KEYS = ["aws"]
        AZURE_KEYS = ["azure"]
        PASSWORD = ["password", "pin"]

    class Organization:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            COMPANY_NAME (list): List of all the patterns for the company name subcategory.
            COMPANY_NUMBER (list): List of all the patterns for the company number subcategory.
            BUSINESS_NUMBER (list): List of all the patterns for the business number subcategory.
        """

        COMPANY_NAME = ["organization"]
        COMPANY_NUMBER = ["company number", "legal entity number"]
        BUSINESS_NUMBER = ["business_number"]

    class DateAndTime:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            DURATION (list): List of all the patterns for the duration subcategory.
            DATE_TIME (list): List of all the patterns for the date time subcategory.
            TIME (list): List of all the patterns for the time subcategory.
            DATE (list): List of all the patterns for the date subcategory.
        """

        DURATION = ["duration", "timeduration", "date_interval"]
        DATE_TIME = ["date_time", "datetime"]
        TIME = ["time"]
        DATE = ["date", "dob"]

    class LocationInformation:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            LOCATION (list): List of all the patterns for the location subcategory.
            ADDRESS (list): List of all the patterns for the address subcategory.
        """

        ADDRESS = ["address", "LOCATION_ADDRESS"]
        LOCATION = [
            "location",
            "LOCATION_CITY",
            "LOCATION_COORDINATE",
            "LOCATION_COUNTRY",
            "LOCATION_STATE",
            "LOCATION_ZIP",
        ]

    class Other:
        """This class contains all the patterns for the subcategories of the anonymization category.

        Constants:
            OTHER (list): List of all the patterns for the other subcategory.
            ANONYMIZED (list): List of all the patterns for the anonymized subcategory.
            NERD (list): List of all the patterns for the nerd subcategory.
            WSD (list): List of all the patterns for the wsd subcategory.
        """

        OTHER = [
            "eu gpu coordinates",
            "u.s. drug enforcement agency (dea) number",
        ]
        ANONYMIZED = ["anonymized"]
        NERD = ["nerd"]
        WSD = ["wsd"]
