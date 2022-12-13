from enum import Enum
import json
import os
from typing import Sequence, Optional

from pydantic import BaseModel, Field, StrictStr, validator

class Language(BaseModel):
    code: StrictStr
    name: StrictStr


class LanguageKey(Enum):
    NAME = 'name'
    CODE = 'code'


def get_info_languages(
    key: LanguageKey,
    value: StrictStr
    ) -> Optional[Language]:
    if value and key:
        subfeature_path = os.path.dirname(os.path.dirname(__file__)) + '/language_detection'
        match_value = lambda x : x.lower() == value.lower()
        with open(f'{subfeature_path}/languages.json', 'r', encoding='utf-8') as file:
            langs = json.load(file)
            lang_idx = next((i for i, val in enumerate(langs) if match_value(val[key.value])), None)
            if lang_idx:
                return langs[lang_idx]
    return None


class InfosLanguageDetectionDataClass(BaseModel):
    language: Language
    confidence: Optional[float]

    @classmethod
    @validator('confidence')
    def normalize_confidence(cls, value):
        return round(value, 2)


class LanguageDetectionDataClass(BaseModel):
    items: Sequence[InfosLanguageDetectionDataClass] = Field(default_factory=list)
