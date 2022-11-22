from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class Items(BaseModel):
    sentiment: StrictStr
    sentiment_rate: Optional[float]


class SentimentAnalysisDataClass(BaseModel):
    items: Sequence[Items] = Field(default_factory=list)
