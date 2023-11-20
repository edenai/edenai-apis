from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class ImageSearchItem(BaseModel):
    image_name: StrictStr


class SearchGetImagesDataClass(BaseModel):
    list_images: Sequence[ImageSearchItem] = Field(default_factory=list)
