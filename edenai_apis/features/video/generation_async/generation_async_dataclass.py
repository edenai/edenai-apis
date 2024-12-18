from pydantic import BaseModel, StrictStr


class GenerationAsyncDataClass(BaseModel):
    video: str
    video_resource_url: StrictStr
