from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class ResumeLocation(BaseModel):
    formatted_location : StrictStr = 'NOT PROVIDED' # Affinda ?
    postal_code : StrictStr # All
    region : StrictStr # All
    country : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    country_code : Optional[StrictStr] = 'NOT PROIDED' # All
    raw_input_location : Optional[StrictStr] # All
    street : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    street_number : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    appartment_number : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    city = StrictStr # All

class ResumeSkill(BaseModel):
    name: StrictStr
    type: Optional[StrictStr]


class ResumeLang(BaseModel):
    name: StrictStr
    code: Optional[StrictStr]


class ResumeWorkExpEntry(BaseModel):
    title: Optional[StrictStr]
    start_date: Optional[StrictStr]
    end_date: Optional[StrictStr]
    company: Optional[StrictStr]
    location: Optional[ResumeLocation]
    description: Optional[StrictStr]
    industry : Optional[StrictStr] = 'NOT PROVIDED' # Hireability


class ResumeWorkExp(BaseModel):
    total_years_experience: Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    entries: Sequence[ResumeWorkExpEntry] = Field(default_factory=list)


class ResumeEducationEntry(BaseModel):
    title: Optional[StrictStr]
    start_date: Optional[StrictStr]
    end_date: Optional[StrictStr]
    location: Optional[ResumeLocation]
    establishment: Optional[StrictStr] # All
    description: Optional[StrictStr]
    gpa : Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    accreditation : Optional[StrictStr] = 'NOT PROVIDED' # Affinda 


class ResumeEducation(BaseModel):
    total_years_education: Optional[int]
    entries: Sequence[ResumeEducationEntry] = Field(default_factory=list)
    
class ResumePersonalName(BaseModel):
    raw_name : StrictStr # all
    first_name : StrictStr # all
    last_name : StrictStr # all
    middle : Optional[StrictStr] = 'NOT PROVIDED' # all
    title : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    prefix : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    sufix : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
class ResumePersonalInfo(BaseModel):
    name : ResumePersonalName
    address: ResumeLocation
    self_summary: Optional[StrictStr] = 'NOT PROVIDED' # all
    objective : Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    date_of_birth : Optional[StrictStr] = 'NOT PROVIDED' # Affinda
    place_of_birth : Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    phones: Sequence[str] = Field(default_factory=list)
    mails: Sequence[str] = Field(default_factory=list)
    urls: Sequence[str] = Field(default_factory=list)
    fax : Sequence[str] = Field(default_factory=list)
    current_profession: Optional[StrictStr] = 'NOT PROVIDED' # ??
    dateOfBirth: Optional[StrictStr] = 'NOT PROVIDED' # affinda
    gender: Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    nationality: Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    martial_status : Optional[StrictStr] = 'NOT PROVIDED' # Hireability
    current_salary : Optional[StrictStr] = 'NOT PROVIDED' # Hireability


class ResumeExtractedData(BaseModel):
    personal_infos: ResumePersonalInfo
    education: ResumeEducation
    work_experience: ResumeWorkExp
    languages: Sequence[ResumeLang] = Field(default_factory=list)
    skills: Sequence[ResumeSkill] = Field(default_factory=list)
    certifications: Sequence[ResumeSkill] = Field(default_factory=list)
    courses: Sequence[ResumeSkill] = Field(default_factory=list)
    publications: Sequence[ResumeSkill] = Field(default_factory=list)
    interests: Sequence[ResumeSkill] = Field(default_factory=list)


class ResumeParserDataClass(BaseModel):
    extracted_data: ResumeExtractedData
