from collections import defaultdict
from typing import Dict, List

import requests

from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import (
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLang,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeLocation,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import (
    ResumeEducation,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class HireabilityApi(ProviderInterface, OcrInterface):
    provider_name = "hireability"

    def __init__(self, api_keys: Dict = {}):
        super().__init__()
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.product_code = self.api_settings["product_code"]
        self.url = "http://processing.resumeparser.com/requestprocessing.html"

    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:
        with open(file, "rb") as file_:
            files = {"document": file_}

            # Generate Api output
            response = requests.post(
                self.url,
                data={
                    "product_code": self.product_code,
                    "document_title": file,
                },
                files=files,
            )
            original_response = response.json()

        # Handle provider error
        if original_response["Results"][0]["HireAbilityJSONResults"][0].get(
            "ProcessingErrors"
        ):
            errors = original_response["Results"][0]["HireAbilityJSONResults"][0][
                "ProcessingErrors"
            ][0]
            raise ProviderException(
                errors["Error"][0]["ErrorMessage"], code=errors["Error"][0]["ErrorCode"]
            )

        infos = original_response["Results"][0]["HireAbilityJSONResults"][0]

        # Resume parser std
        default_dict = defaultdict(lambda: None)
        # 1. Personal informations
        # 1.1 Resume name
        personal_names = ResumePersonalName(
            first_name=infos.get("GivenName", ""),
            last_name=infos.get("FamilyName", ""),
            raw_name=infos.get("FormattedName"),
            middle=None,
            prefix=None,
            sufix=None,
            title=None,
        )
        # 1.2 Address
        address = ResumeLocation(
            country_code=infos.get("CountryCode"),
            region=infos.get("CountrySubDivisionCode"),
            postal_code=infos.get("PostalCode"),
            street=infos.get("AddressLine"),
            city=infos.get("CityName", ""),
            formatted_location=None,
            street_number=None,
            appartment_number=None,
            raw_input_location=None,
            country=None,
        )
        # 1.3 Others
        personal_infos = ResumePersonalInfo(
            name=personal_names,
            address=address,
            phones=[
                phone.get("Number")
                for phone in infos.get("Phone", default_dict)
                if phone.get("Number")
            ],
            mails=[
                mail.get("Address")
                for mail in infos.get("Email", default_dict)
                if mail.get("Address")
            ],
            urls=[],
            gender=infos.get("Gender"),
            date_of_birth=infos.get("DateOfBirth"),
            place_of_birth=infos.get("PlaceOfBirth"),
            nationality=infos.get("Nationality"),
            martial_status=infos.get("MaritalStatus"),
            objective=infos.get("Objective"),
            current_salary=infos.get("Salary"),
            self_summary=None,
            current_profession=None,
        )

        # 2 Education
        edu_entries: List[ResumeEducationEntry] = []
        for i in infos.get("EducationOrganizationAttendance") or []:
            title = i.get("EducationLevel", [{}])[0].get(
                "Name"
            )  # value of MajorProgramName can be None
            description = (i.get("MajorProgramName") or [None])[0]
            if isinstance(i.get("ReferenceLocation"), dict):
                location = ResumeLocation(
                    country_code=i.get("ReferenceLocation", {}).get("CountryCode"),
                    region=i.get("ReferenceLocation", {}).get("CountrySubDivisionCode"),
                    city=i.get("ReferenceLocation", {}).get("CityName", ""),
                    formatted_location=None,
                    postal_code=None,
                    country=None,
                    raw_input_location=None,
                    street=None,
                    street_number=None,
                    appartment_number=None,
                )
            else:
                location = ResumeLocation(
                    country_code=None,
                    region=None,
                    city=None,
                    formatted_location=None,
                    postal_code=None,
                    country=None,
                    raw_input_location=None,
                    street=None,
                    street_number=None,
                    appartment_number=None,
                )
            edu_entries.append(
                ResumeEducationEntry(
                    start_date=i.get("AttendanceStartDate"),
                    end_date=i.get("AttendanceEndDate"),
                    establishment=i.get("School"),
                    location=location,
                    title=title,
                    description=description,
                    gpa=i.get("EducationScore", [""])[0],
                    accreditation=None,
                )
            )
        edu = ResumeEducation(entries=edu_entries, total_years_education=None)

        # 3 Work experience
        work_entries = []
        for i in infos.get("PositionHistory", []):
            work_location = ResumeLocation(
                country_code=(i.get("ReferenceLocation") or {}).get("CountryCode"),
                region=(i.get("ReferenceLocation") or {}).get("CountrySubDivisionCode"),
                city=(i.get("ReferenceLocation") or {}).get("CityName", ""),
                formatted_location=None,
                postal_code=None,
                country=None,
                raw_input_location=None,
                street=None,
                street_number=None,
                appartment_number=None,
            )
            work_entries.append(
                ResumeWorkExpEntry(
                    title=i.get("PositionTitle"),
                    company=i.get("Employer"),
                    start_date=i.get("StartDate"),
                    end_date=i.get("EndDate"),
                    description=i.get("Description"),
                    location=work_location,
                    industry=(i.get("Industry", {}) or {}).get("Name"),
                )
            )
        duration = infos.get("TotalYearsOfExperience")
        work = ResumeWorkExp(total_years_experience=duration, entries=work_entries)

        # 4 Others
        skills = []
        for i in infos.get("PersonCompetency") or []:
            skill = i.get("CompetencyName")
            skill_type = i.get("CompetencyLevel")
            skills.append(ResumeSkill(name=skill, type=skill_type))

        languages = [
            ResumeLang(name=i.get("LanguageCode"), code=None)
            for i in infos.get("Languages", [])
        ]
        certifications = [
            ResumeSkill(name=i.get("CertificationName"), type=None)
            for i in infos.get("Certification", [])
        ]

        std = ResumeParserDataClass(
            extracted_data=ResumeExtractedData(
                personal_infos=personal_infos,
                education=edu,
                work_experience=work,
                languages=languages,
                skills=skills,
                certifications=certifications,
            )
        )

        result = ResponseType[ResumeParserDataClass](
            original_response=original_response, standardized_response=std
        )
        return result
