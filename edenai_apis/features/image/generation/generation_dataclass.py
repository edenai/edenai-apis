from typing import Sequence

from pydantic import BaseModel, Field, StrictStr


class GeneratedImageDataClass(BaseModel):
    image: str
    image_resource_url: StrictStr


class GenerationDataClass(BaseModel):
    items: Sequence[GeneratedImageDataClass] = Field(default_factory=list)
