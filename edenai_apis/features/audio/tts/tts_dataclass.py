from pydantic import BaseModel, StrictStr


class TtsDataClass(BaseModel):
    audio_resource_url: StrictStr
