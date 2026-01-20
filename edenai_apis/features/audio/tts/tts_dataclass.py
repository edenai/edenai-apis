from pydantic import BaseModel, StrictStr


class TtsDataClass(BaseModel):
    audio: StrictStr
    audio_resource_url: StrictStr
