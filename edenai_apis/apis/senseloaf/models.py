from typing import Dict
from pydantic import StrictStr, BaseModel


class ResponseData(BaseModel):
    response: Dict
    response_code: StrictStr
    response_type: StrictStr
