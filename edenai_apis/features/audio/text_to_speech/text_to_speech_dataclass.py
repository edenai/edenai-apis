from typing import Dict
from pydantic import BaseModel, StrictStr


class TextToSpeechDataClass(BaseModel):
    audio: StrictStr
    voice_type: int

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["audio"]
