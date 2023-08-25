from pydantic import BaseModel, StrictStr


class AutomlClassificationCreateEndpoint(BaseModel):
    endpoint_id: StrictStr
