from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class BoundingBox(BaseModel):
    top: float
    left: float
    width: float
    height: float


class FieldIdentityParserDataClass(BaseModel):
    value: StrictStr
    bounding_box: BoundingBox
    confidence: float


class InfosIdentityParserDataClass(BaseModel):
    last_name: FieldIdentityParserDataClass
    given_names: Sequence[StrictStr] = Field(default_factory=list)
    date_of_birth: FieldIdentityParserDataClass
    birth_place: FieldIdentityParserDataClass
    country: FieldIdentityParserDataClass
    address: FieldIdentityParserDataClass
    nationality: FieldIdentityParserDataClass
    gender: FieldIdentityParserDataClass
    issuance_date: FieldIdentityParserDataClass
    expire_date: FieldIdentityParserDataClass
    mrz: FieldIdentityParserDataClass
    document_id: FieldIdentityParserDataClass
    age: Optional[FieldIdentityParserDataClass]
    document_type: Optional[StrictStr]


class IdentityParserDataClass(BaseModel):
    extracted_data: Sequence[InfosIdentityParserDataClass] = Field(default_factory=list)