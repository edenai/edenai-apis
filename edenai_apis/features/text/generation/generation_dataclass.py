from typing import Dict

from pydantic import BaseModel, StrictStr


class GenerationDataClass(BaseModel):
    generated_text: StrictStr

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["generated_text"]
