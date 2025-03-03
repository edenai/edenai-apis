from typing import Any, Dict, List, Tuple

from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeLocation,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeExtractedData,
    ResumeLang,
    ResumeParserDataClass,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)
from edenai_apis.utils.conversion import _convert_dictionary_to_date_string
from .models import ResponseData


class ResumeMapper:

    __original_response: Dict[str, Any]
    __standard_response: Dict[str, Any]

    def __init__(self, original_response: ResponseData) -> None:
        self.__original_response = original_response.response["result"][0]["content"]
        self.__standard_response = {}
        self.map_response()

    def __map_name(self) -> ResumePersonalName:
        name = self.__original_response.get("contactInfo", {})
        raw_name = (
            (
                name.get("title")
                + " "
                + name.get("firstName")
                + " "
                + name.get("middleName")
                + " "
                + name.get("lastName")
            )
            .strip(" ")
            .replace("  ", " ")
        )
        return ResumePersonalName(
            raw_name=raw_name,
            first_name=name.get("firstName"),
            last_name=name.get("lastName"),
            middle=name.get("middleName"),
            title=name.get("title"),
            sufix=None,
            prefix=None,
        )

    def __map_location(self) -> ResumeLocation:
        location = self.__original_response["contactMethod"]["PostalAddress"][
            "Location"
        ]
        raw_location = (
            (
                location["Area"]
                + " "
                + location["City"]
                + " "
                + location["State"]
                + " "
                + location["Country"]
                + " "
                + location["Zip"]
            )
            .strip(" ")
            .replace("  ", " ")
        )
        return ResumeLocation(
            raw_input_location=None,
            postal_code=location.get("Zip"),
            region=location.get("State"),
            country_code=location.get("CountryCode"),
            country=location.get("Country"),
            appartment_number=None,
            city=location.get("City"),
            street=location.get("Area"),
            street_number=None,
            formatted_location=raw_location,
        )

    def __get_summary(self) -> str:
        summaries = [
            i["sectionText"]
            for i in self.__original_response["sections"]
            if i["sectionName"] == "Summary"
        ]
        return " ".join(summaries)

    def __get_objective(self) -> str:
        objectives = [
            i["sectionText"]
            for i in self.__original_response["sections"]
            if i["sectionName"] == "Objective"
        ]
        return " ".join(objectives)

    def __map_pi(self) -> ResumePersonalInfo:
        self.__standard_response["personal_infos"] = ResumePersonalInfo(
            name=self.__map_name(),
            address=self.__map_location(),
            self_summary=self.__get_summary(),
            objective=self.__get_objective(),
            date_of_birth=None,
            place_of_birth=None,
            phones=self.__original_response["contactMethod"]["Telephone"],
            mails=self.__original_response["contactMethod"]["ContactEmailid"],
            urls=self.__original_response["socialLinks"]["urls"],
            fax=[],
            current_profession=self.__original_response["experience"]["CurrentJobRole"],
            gender=None,
            nationality=None,
            martial_status=None,
            current_salary=None,
        )
        return self.__standard_response["personal_infos"]

    def __map_education(self) -> ResumeEducation:
        edu_entries: List[ResumeEducationEntry] = []
        for i in self.__original_response["education"]["EducationSection"]:
            location = ResumeLocation(
                formatted_location=i["Location"],
                raw_input_location=None,
                postal_code=None,
                region=None,
                country_code=None,
                country=None,
                appartment_number=None,
                city=None,
                street=None,
                street_number=None,
            )
            start_date = _convert_dictionary_to_date_string(i["EduStartDate"])
            end_date = _convert_dictionary_to_date_string(i["EduEndDate"])
            edu_entries.append(
                ResumeEducationEntry(
                    title=i["Degree"],
                    start_date=start_date,
                    end_date=end_date,
                    location=location,
                    establishment=i["Collegename"],
                    description=None,
                    gpa=i["Grade"],
                    accreditation=i["Major"],
                )
            )
        self.__standard_response["education"] = ResumeEducation(
            entries=edu_entries, total_years_education=None
        )
        return self.__standard_response["education"]

    def __map_experience(self) -> ResumeWorkExp:
        work_entries: List[ResumeWorkExpEntry] = []
        for i in self.__original_response["experience"]["ExperienceSection"]:
            location = ResumeLocation(
                formatted_location=i["ExpLocation"],
                raw_input_location=None,
                postal_code=None,
                region=None,
                country_code=None,
                country=None,
                appartment_number=None,
                city=None,
                street=None,
                street_number=None,
            )
            start_date = _convert_dictionary_to_date_string(i["ExpStartDate"])
            end_date = _convert_dictionary_to_date_string(i["ExpEndDate"])
            work_entries.append(
                ResumeWorkExpEntry(
                    title=i["JobTitle"],
                    company=i["CompanyName"],
                    start_date=start_date,
                    end_date=end_date,
                    location=location,
                    industry=None,
                    description=i["Description"],
                )
            )
        self.__standard_response["work_experience"] = ResumeWorkExp(
            total_years_experience=str(
                self.__original_response["experience"]["TotalExperience"]
            ),
            entries=work_entries,
        )

        return self.__standard_response["work_experience"]

    def __map_skills(self) -> List[ResumeSkill]:
        self.__standard_response["skills"] = []
        for i in self.__original_response.get("skills", []) or []:
            if "Language" in i.get("SkillType", []):
                continue
            skill_name = i.get("SkillName")
            skill_type = i.get("SkillType", [])
            skill_type = None if len(skill_type) == 0 else skill_type[0]
            self.__standard_response["skills"].append(
                ResumeSkill(name=skill_name, type=skill_type)
            )
        return self.__standard_response["skills"]

    def __map_others(self) -> Tuple:
        skills = self.__original_response.get("skills", [])
        languages = [
            ResumeLang(name=i.get("SkillName"), code=None)
            for i in skills
            if "Language" in i.get("SkillType", [])
        ]
        self.__standard_response["languages"] = languages
        self.__standard_response["certifications"] = []
        self.__standard_response["publications"] = []

        return (
            self.__standard_response["languages"],
            self.__standard_response["certifications"],
            self.__standard_response["publications"],
        )

    def map_response(self) -> None:
        self.__map_pi()
        self.__map_education()
        self.__map_experience()
        self.__map_skills()
        self.__map_others()

    def standard_response(self):
        return ResumeParserDataClass(
            extracted_data=ResumeExtractedData(**self.__standard_response)
        )

    def original_response(self):
        return self.__original_response
