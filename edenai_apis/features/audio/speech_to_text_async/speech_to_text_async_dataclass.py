from typing import Sequence, Optional

from pydantic import BaseModel, StrictStr, Field


class SpeechDiarizationEntry(BaseModel):
    segment: StrictStr
    start_time: Optional[StrictStr]
    end_time: Optional[StrictStr]
    speaker: int
    confidence: Optional[float]


class SpeechDiarization(BaseModel):
    total_speakers: int
    entries: Sequence[SpeechDiarizationEntry] = Field(default_factory=list)
    error_message: Optional[str] = None


class SpeechToTextAsyncDataClass(BaseModel):
    text: StrictStr
    diarization: SpeechDiarization
