from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class VariationImageDataClass(BaseModel):
    image: str
    image_resource_url: StrictStr


class VariationDataClass(BaseModel):
    items: Sequence[VariationImageDataClass] = Field(default_factory=list)
