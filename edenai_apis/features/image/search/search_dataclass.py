from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class ImageItem(BaseModel):
    image_name: StrictStr
    score: float


class SearchDataClass(BaseModel):
    items: Sequence[ImageItem] = Field(default_factory=list)
