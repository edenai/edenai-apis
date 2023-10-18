from pydantic import BaseModel

class AnonymizationDataClass(BaseModel):
    document: str