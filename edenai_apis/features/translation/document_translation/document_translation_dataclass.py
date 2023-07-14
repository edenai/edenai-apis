from pydantic import BaseModel, StrictStr
from typing import Dict


class DocumentTranslationDataClass(BaseModel):
    file: str
    document_resource_url: StrictStr

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["file"]
