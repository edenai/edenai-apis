from pydantic import BaseModel


class AnonymizationAsyncDataClass(BaseModel):
    document: str
    document_url: str
