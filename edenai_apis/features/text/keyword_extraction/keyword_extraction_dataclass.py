from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class InfosKeywordExtractionDataClass(BaseModel):
    keyword: StrictStr
    importance: Optional[float]


class KeywordExtractionDataClass(BaseModel):
    items: Sequence[InfosKeywordExtractionDataClass] = Field(default_factory=list)
