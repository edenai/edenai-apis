from pydantic import BaseModel, StrictStr
from typing import Dict


class DocumentTranslationDataClass(BaseModel):
    file: str
    
    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["file"]
