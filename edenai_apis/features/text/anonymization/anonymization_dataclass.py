from typing import Dict
from pydantic import BaseModel, StrictStr


class AnonymizationDataClass(BaseModel):
    result: StrictStr

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["result"]
