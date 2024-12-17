from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class GenerationAsyncDataclass(BaseModel):
    video: str
    video_resource_url: StrictStr
