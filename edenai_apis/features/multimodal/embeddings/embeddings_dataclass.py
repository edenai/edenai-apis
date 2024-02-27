from typing import List, Optional

from pydantic import BaseModel, Field


class VideoEmbeddingModel(BaseModel):
    embedding: List[float] = Field(default_factory=list)
    start_offset: Optional[float] = None
    end_offset: Optional[float] = None


class EmbeddingModel(BaseModel):
    text_embedding: List[float] = Field(default_factory=list)
    image_embedding: List[float] = Field(default_factory=list)
    video_embeddings: List[VideoEmbeddingModel] = Field(default_factory=list)


class EmbeddingsDataClass(BaseModel):
    items: List[EmbeddingModel] = Field(default_factory=list)
