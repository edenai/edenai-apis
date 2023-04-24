import datetime
from dateutil import parser
from enum import Enum
import json
import os
from typing import Optional, Sequence, Any

from pydantic import BaseModel, Field, StrictStr, validator


def format_date(value):
    if not value:
        return None
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    value = parser.parse(value)
    return value.strftime('%Y-%m-%d')


class Country(BaseModel):
    name: Optional[StrictStr]
    alpha2: Optional[StrictStr]
    alpha3: Optional[StrictStr]
    confidence: Optional[float]


class InfoCountry(Enum):
    NAME = 'name'
    ALPHA2 = 'alpha2'
    ALPHA3 = 'alpha3'


def get_info_country(
    key: InfoCountry,
    value: StrictStr
) -> Optional[Country]:
    feature_path = os.path.dirname(os.path.dirname(__file__))

    if not value or not key:
        return None

    with open(f'{feature_path}/identity_parser/countries.json', 'r', encoding='utf-8') as f:
        countries = json.load(f)
        country_idx = next((index for (index, country) in enumerate(
            countries) if country[key.value].lower() == value.lower()), None)
        if country_idx:
            return countries[country_idx]
        print(f"{key}: {value} not found")
    return None


class ItemIdentityParserDataClass(BaseModel):
    value: Optional[StrictStr]
    confidence: Optional[float]


class InfosIdentityParserDataClass(BaseModel):
    last_name: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    given_names: Sequence[ItemIdentityParserDataClass] = Field(
        default_factory=list)
    birth_place: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    birth_date: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    issuance_date: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    expire_date: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    document_id: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    issuing_state: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    address: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    age: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    country: Optional[Country] = Country()
    document_type: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    gender: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    image_id: Sequence[ItemIdentityParserDataClass] = Field(
        default_factory=list)
    image_signature: Sequence[ItemIdentityParserDataClass] = Field(
        default_factory=list)
    mrz: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)
    nationality: Optional[ItemIdentityParserDataClass] = Field(
        default_factory=ItemIdentityParserDataClass)

    @validator('last_name')
    def to_uppercase(cls, value):
        value.value = value.value.upper() if value.value else None
        return value

    @validator('given_names', each_item=True)
    def list_to_title(cls, value):
        value.value = value.value.title() if value.value else None
        return value

    @validator('birth_place', 'nationality', 'issuing_state', 'document_type')
    def to_title(cls, value):
        value.value = value.value.title() if value.value else None
        return value

    @validator('expire_date', 'issuance_date', 'birth_date')
    def date_validator(cls, value):
        if not value.value:
            return None
        try:
            datetime.datetime.strptime(value.value, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        return value


class IdentityParserDataClass(BaseModel):
    extracted_data: Sequence[InfosIdentityParserDataClass] = Field(
        default_factory=list)
