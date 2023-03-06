from pydantic import BaseModel, Field, StrictStr
from typing import Sequence


class GeneratedImageDataClass(BaseModel):
    image: str
    image_resource_url: StrictStr
    
class GenerationDataClass(BaseModel):
    items: Sequence[GeneratedImageDataClass] = Field(default_factory=list)

