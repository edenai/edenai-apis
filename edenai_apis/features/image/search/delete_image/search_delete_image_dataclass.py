from pydantic import BaseModel


class SearchDeleteImageDataClass(BaseModel):
    status: str
