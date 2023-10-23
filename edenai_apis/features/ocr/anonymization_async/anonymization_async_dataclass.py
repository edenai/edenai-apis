from pydantic import BaseModel

class AnonymizationAsyncDataClass(BaseModel):
    document: str