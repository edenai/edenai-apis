from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLocation,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeLang,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)

from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    MicrModel,
    ItemBankCheckParsingDataClass,
)


def extracta_resume_parser(original_response: dict) -> ResumeParserDataClass:
    # personal infos
    name = ResumePersonalName(
        raw_name=original_response.get("name", None),
        first_name=original_response.get("first_name", None),
        last_name=original_response.get("last_name", None),
        middle=None,
        title=None,
        prefix=None,
        sufix=None,
    )

    address = ResumeLocation(
        formatted_location=original_response.get("address", None),
        postal_code=None,
        region=None,
        country=None,
        country_code=None,
        raw_input_location=None,
        street=None,
        street_number=None,
        appartment_number=None,
        city=None,
    )

    phones = []
    for phone in original_response.get("phones", []):
        phones.append(phone)

    mails = []
    for mail in original_response.get("mails", []):
        mails.append(mail)

    urls = []
    for url in original_response.get("urls", []):
        urls.append(url)

    personal_infos = ResumePersonalInfo(
        name=name,
        address=address,
        self_summary=original_response.get("self_summary", None),
        objective=None,
        date_of_birth=None,
        place_of_birth=None,
        phones=phones,
        mails=mails,
        urls=urls,
        fax=[],
        current_profession=None,
        gender=None,
        nationality=None,
        martial_status=None,
        current_salary=None,
    )

    # education
    education_entries = []
    for edu in original_response.get("education", []):
        education_address = ResumeLocation(
            formatted_location=edu.get("address", None),
            postal_code=None,
            region=None,
            country=None,
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=None,
        )
        education_entries.append(
            ResumeEducationEntry(
                title=edu.get("title", None),
                start_date=edu.get("start_date", None),
                end_date=edu.get("end_date", None),
                location=education_address,
                establishment=edu.get("establishment", None),
                description=edu.get("description", None),
                gpa=edu.get("gpa", None),
                accreditation=edu.get("accreditation", None),
            )
        )
    education = ResumeEducation(
        total_years_education=None,
        entries=education_entries,
    )

    # work experience
    work_experience_entries = []
    for work in original_response.get("work_experience", []):
        work_address = ResumeLocation(
            formatted_location=work.get("address", None),
            postal_code=None,
            region=None,
            country=None,
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=None,
        )
        work_experience_entries.append(
            ResumeWorkExpEntry(
                title=work.get("title", None),
                start_date=work.get("start_date", None),
                end_date=work.get("end_date", None),
                company=work.get("company_name", None),
                location=work_address,
                description=work.get("description", None),
                industry=work.get("industry", None),
            )
        )
    work_experience = ResumeWorkExp(
        total_years_experience=None,
        entries=work_experience_entries,
    )

    # language
    languages = []
    for lang in original_response.get("languages", []):
        languages.append(
            ResumeLang(name=lang.get("name", None), code=lang.get("code", None))
        )

    # skills
    skills = []
    for skill in original_response.get("skills", []):
        skills.append(
            ResumeSkill(name=skill.get("name", None), type=skill.get("type", None))
        )

    # certifications
    certifications = []
    for cert in original_response.get("certifications", []):
        certifications.append(
            ResumeSkill(name=cert.get("name", None), type=cert.get("type", None))
        )

    # courses
    courses = []
    for course in original_response.get("courses", []):
        courses.append(
            ResumeSkill(name=course.get("name", None), type=course.get("type", None))
        )

    # publications
    publications = []
    for publication in original_response.get("publications", []):
        publications.append(
            ResumeSkill(
                name=publication.get("name", None), type=publication.get("type", None)
            )
        )

    # interests
    interests = []
    for interest in original_response.get("interests", []):
        interests.append(
            ResumeSkill(
                name=interest.get("name", None), type=interest.get("type", None)
            )
        )

    # extracted data
    extracted_data = ResumeExtractedData(
        personal_infos=personal_infos,
        education=education,
        work_experience=work_experience,
        languages=languages,
        skills=skills,
        certifications=certifications,
        courses=courses,
        publications=publications,
        interests=interests,
    )

    return ResumeParserDataClass(extracted_data=extracted_data)


def extracta_bank_check_parsing(original_response: dict) -> ResumeParserDataClass:
    # extracted data
    extracted_data = [
        ItemBankCheckParsingDataClass(
            amount=original_response.get("amount", None),
            amount_text=original_response.get("amount_text", None),
            bank_address=original_response.get("bank_address", None),
            bank_name=original_response.get("bank_name", None),
            date=original_response.get("date", None),
            memo=original_response.get("memo", None),
            payer_address=original_response.get("payer_address", None),
            payer_name=original_response.get("payer_name", None),
            receiver_address=original_response.get("receiver_address", None),
            receiver_name=original_response.get("receiver_name", None),
            currency=original_response.get("currency", None),
            micr=MicrModel(
                raw=original_response.get("micr", {}).get("micr_raw", None),
                account_number=original_response.get("micr", {}).get(
                    "account_number", None
                ),
                routing_number=original_response.get("micr", {}).get(
                    "routing_number", None
                ),
                serial_number=original_response.get("micr", {}).get(
                    "serial_number", None
                ),
                check_number=original_response.get("micr", {}).get(
                    "check_number", None
                ),
            ),
        )
    ]

    return BankCheckParsingDataClass(extracted_data=extracted_data)
