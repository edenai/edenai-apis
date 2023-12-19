from typing import Dict, Any

from pydantic import BaseModel, StrictStr, field_validator


class AutomaticTranslationDataClass(BaseModel):
    text: StrictStr

    @classmethod
    @field_validator("text", mode="before")
    def check_text(cls, value: StrictStr) -> StrictStr:
        if not value:
            value = ""
        return value

    @staticmethod
    def direct_response(api_response: Dict[Any, Any]) -> StrictStr:
        return api_response["text"]
