from pydantic import BaseModel, StrictStr, Field
from typing import Sequence




class SpeechDiarizationEntry(BaseModel):
    text : StrictStr
    start_time: StrictStr
    end_time: StrictStr
    speaker_tag: int
class SpeechDiarization(BaseModel):
    total_speakers : int
    entries: Sequence[SpeechDiarizationEntry] = Field(default_factory=list)
class SpeechToTextAsyncDataClass(BaseModel):
    text: StrictStr
    diarization: SpeechDiarization



