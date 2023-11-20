from pydantic import BaseModel


class SearchGetImageDataClass(BaseModel):
    image: bytes
