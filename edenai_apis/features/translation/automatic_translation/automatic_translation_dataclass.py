from typing import Dict
from pydantic import BaseModel, StrictStr




class AutomaticTranslationDataClass(BaseModel):
    text: StrictStr

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["text"]
