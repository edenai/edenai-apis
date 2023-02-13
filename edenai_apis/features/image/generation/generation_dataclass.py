from pydantic import BaseModel, Field
from typing import Sequence


class GeneratedImageDataClass(BaseModel):
    image: str
    
class GenerationDataClass(BaseModel):
    items: Sequence[GeneratedImageDataClass] = Field(default_factory=list)

