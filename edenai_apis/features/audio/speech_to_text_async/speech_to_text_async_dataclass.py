from pydantic import BaseModel, StrictStr


class SpeechToTextAsyncDataClass(BaseModel):
    text: StrictStr
