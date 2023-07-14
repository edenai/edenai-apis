from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class InfosCustomNamedEntityRecognitionDataClass(BaseModel):
    entity: StrictStr
    category: StrictStr


class CustomNamedEntityRecognitionDataClass(BaseModel):
    items: Sequence[InfosCustomNamedEntityRecognitionDataClass] = Field(
        default_factory=list
    )
