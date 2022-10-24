from pydantic import BaseModel, StrictStr

class AnonymizationDataClass(BaseModel):
    image: StrictStr
