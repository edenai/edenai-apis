from pydantic import BaseModel, StrictStr, Field
from typing import Sequence, Optional




class SpeechDiarizationEntry(BaseModel):
    segment : StrictStr
    start_time: StrictStr
    end_time: StrictStr
    speaker: int
    confidence: Optional[float]


class SpeechDiarization(BaseModel):
    total_speakers : int
    entries: Sequence[SpeechDiarizationEntry] = Field(default_factory=list)


class SpeechToTextAsyncDataClass(BaseModel):
    text: StrictStr
    diarization: SpeechDiarization



