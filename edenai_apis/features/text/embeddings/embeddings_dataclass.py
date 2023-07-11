from typing import Sequence
from pydantic import BaseModel, Field


class EmbeddingDataClass(BaseModel):
    embedding: Sequence[float]


class EmbeddingsDataClass(BaseModel):
    items: Sequence[EmbeddingDataClass] = Field(default_factory=list)
