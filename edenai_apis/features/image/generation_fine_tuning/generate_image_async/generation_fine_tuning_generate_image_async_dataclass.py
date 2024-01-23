from typing import List

from pydantic import BaseModel


class GenerationFineTuningGenerateImageAsyncDataClass(BaseModel):
    images_url: List[str]
