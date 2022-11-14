from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class ResumeSkill(BaseModel):
    name: StrictStr
    type: Optional[StrictStr]


class ResumeLang(BaseModel):
    name: StrictStr
    value: Optional[StrictStr]


class ResumeWorkExpEntry(BaseModel):
    title: Optional[StrictStr]
    start_date: Optional[StrictStr]
    end_date: Optional[StrictStr]
    company: Optional[StrictStr]
    location: Optional[StrictStr]
    description: Optional[StrictStr]


class ResumeWorkExp(BaseModel):
    total_years_experience: int
    entries: Sequence[ResumeWorkExpEntry] = Field(default_factory=list)


class ResumeEducationEntry(BaseModel):
    title: Optional[StrictStr]
    start_date: Optional[StrictStr]
    end_date: Optional[StrictStr]
    location: Optional[StrictStr]
    establishment: Optional[StrictStr]
    description: Optional[StrictStr]


class ResumeEducation(BaseModel):
    total_years_education: Optional[int]
    entries: Sequence[ResumeEducationEntry] = Field(default_factory=list)


class ResumePersonalInfo(BaseModel):
    first_name: StrictStr
    last_name: StrictStr
    address: Optional[StrictStr]
    self_summary: StrictStr
    phones: Sequence[str] = Field(default_factory=list)
    mails: Sequence[str] = Field(default_factory=list)
    urls: Sequence[str] = Field(default_factory=list)
    current_profession: Optional[StrictStr]
    driving_licence: Optional[StrictStr]
    gender: Optional[float]


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
