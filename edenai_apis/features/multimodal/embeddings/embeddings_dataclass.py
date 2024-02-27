from typing import List

from pydantic import BaseModel, Field


class EmbeddingModel(BaseModel):
    text_embedding: List[float] = Field(default_factory=list)
    image_embedding: List[float] = Field(default_factory=list)
    video_embedding: List[float] = Field(default_factory=list)


class EmbeddingsDataClass(BaseModel):
    items: List[EmbeddingModel] = Field(default_factory=list)
