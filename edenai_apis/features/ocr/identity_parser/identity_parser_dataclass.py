from datetime import datetime
import os
import json
from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr, validator


class Country(BaseModel):
    name: Optional[StrictStr]
    alpha2: Optional[StrictStr]
    alpha3: Optional[StrictStr]

class InfoCountry(enumerate):
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

    with open(f'{feature_path}/identity_parser/countries.json', 'r') as f:
        countries = json.load(f)
        country_idx = next((index for (index, country) in enumerate(countries) if country[key].lower()== value.lower()), None)
        if country_idx:
            return countries[country_idx]
        print(f"{key}: {value} not found")
    return None

def format_date(value, format):
    if not value or not format:
        return None
    return datetime.strptime(value, format).strftime('%Y-%m-%d')

class InfosIdentityParserDataClass(BaseModel):
    last_name: Optional[StrictStr]
    given_names: Sequence[StrictStr] = Field(default_factory=list)
    birth_place: Optional[StrictStr]
    birth_date: Optional[StrictStr]
    issuance_date: Optional[StrictStr]
    expire_date: Optional[StrictStr]
    document_id: Optional[StrictStr]
    issuing_state: Optional[StrictStr]
    address: Optional[StrictStr]
    age: Optional[int]
    country: Optional[Country] = Country()
    document_type: Optional[StrictStr]
    gender: Optional[StrictStr]
    image_id: Sequence[StrictStr] = Field(default_factory=list)
    image_signature: Sequence[StrictStr] = Field(default_factory=list)
    mrz: Optional[StrictStr]
    nationality: Optional[StrictStr]

    @validator('last_name')
    def to_uppercase(cls, value):
        return value.upper() if value else value

    @validator('given_names', each_item=True)
    def list_to_title(cls, value):
        return value.title() if value else value

    @validator('birth_place', 'nationality', 'issuing_state', 'document_type')
    def to_title(cls, value):
        return value.title() if value else value

    @validator('expire_date', 'issuance_date', 'birth_date')
    def date_validator(cls, value):
        if not value:
            return None
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        return value

    
class IdentityParserDataClass(BaseModel):
    extracted_data: Sequence[InfosIdentityParserDataClass] = Field(default_factory=list)