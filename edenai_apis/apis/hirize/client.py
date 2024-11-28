import requests
from edenai_apis.utils.types import ResponseType

from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLocation,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)
from edenai_apis.utils.exception import ProviderException

from typing import List
from json import JSONDecodeError

class Client:

    """This class are used to simplify the usage of Hirize api
    Postman collection
        https://documenter.getpostman.com/view/25317643/2s8ZDSb5Ha
    Constants:
        BASE_URL (str): base url for the Hirize (https://connect.hirize.hr/api/public)
    Methods:
        parse_resume(file: str): This function are used to parse a resume file
    """

    __api_keys: str
    __header: str
    __url: str
    __data: str

    def __init__(self, api_keys, header, data, url) -> None:
        self.__api_keys = api_keys
        self.__header = header
        self.__data = data
        self.__url = url
        self.headers = {"Authorization": f"Bearer {self.__api_keys}"}


    def ocr_resume_parser(self) -> ResponseType[ResumeParserDataClass]:


        data = requests.request("POST", self.__url, headers=self.__header, data=self.__data)

        try :

            data = data.json()
            original_response = data.get("data", {}).get("result", {})
            infos = original_response.get("basic_info", {})
            first_name = infos.get("name", "").split(" ")[0]
            last_name = infos.get("name", "").split(" ")[1:]
            last_name = " ".join(last_name)
            contact_info = original_response.get("contact_info", {})

            phone = contact_info.get("phone_number", [])

            personal_names = ResumePersonalName(
                first_name=first_name,
                last_name=last_name,
                raw_name=infos.get("name", ""),
                middle=None,
                prefix=None,
                sufix=None,
                title=None,
            )
            # 1.2 Address
            address = ResumeLocation(
                country_code=None,
                region=None,
                postal_code=contact_info.get("zip_code"),
                city=contact_info.get("city"),
                street=infos.get("street_address"),
                formatted_location=None,
                street_number=None,
                appartment_number=None,
                raw_input_location=None,
                country=contact_info.get("country")
            )
            # 1.3 Others
            personal_infos = ResumePersonalInfo(
                name=personal_names,
                address=address,
                phones=[phone],
                mails=[
                         contact_info.get("email"),
                      ],
                urls=[infos.get("links")],
                gender=None,
                date_of_birth=None,
                place_of_birth=None,
                nationality=None,
                martial_status=None,
                objective=None,
                current_salary=None,
                self_summary=None,
                current_profession=None,
            )

            # 2 Education
            edu_entries: List[ResumeEducationEntry] = []

            for i in original_response.get("education") or []:
                title = i.get("education_level", "")
                description = i.get("major", "")
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
                        start_date=None,
                        end_date=i.get("graduation_year"),
                        establishment=i.get("school_name"),
                        location=location,
                        title=title,
                        description=description,
                        gpa=None,
                        accreditation=None,
                    )
                )
            edu = ResumeEducation(entries=edu_entries, total_years_education=None)

            # 3 Work experience
            work_entries = []
            for i in original_response.get("work_experience", []).get("works") or []:
                work_location = ResumeLocation(
                    country_code=None,
                    region=None,
                    city= None,
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
                        title=i.get("job_title"),
                        company=i.get("company"),
                        start_date=i.get("start_date"),
                        end_date=i.get("end_date"),
                        description="",
                        location=work_location,
                        industry="",
                    )
                )

            duration = original_response.get("total_work_experience")
            work = ResumeWorkExp(total_years_experience=duration, entries=work_entries)

            # 4 Others
            skills = []
            for i in original_response.get("skills") or []:
                skill = i
                skill_type = None
                skills.append(ResumeSkill(name=skill, type=skill_type))


            std = ResumeParserDataClass(
                extracted_data=ResumeExtractedData(
                    personal_infos=personal_infos,
                    education=edu,
                    work_experience=work,
                    languages=None,
                    skills=skills,
                    certifications=None,
                )
            )

            result = ResponseType[ResumeParserDataClass](
                original_response=original_response, standardized_response=std
            )

            return result

        except requests.exceptions.RequestException as exc:

            raise ProviderException(
                message=f"{exc}",
                code=data.status_code
            ) from exc

        except JSONDecodeError:
            raise ProviderException(
                message="Internal server error", code=data.status_code
            )