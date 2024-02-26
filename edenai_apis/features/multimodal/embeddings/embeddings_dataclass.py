from typing import List

from pydantic import BaseModel, Field


class EmbeddingsDataClass(BaseModel):
    embeddings: List[float] = Field(default_factory=list)
