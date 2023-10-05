from enum import Enum
from typing import Sequence
from pydantic import BaseModel, Field


class EmotionEnum(Enum):
    SADNESS = "Sadness"
    JOY = "Joy"
    ANGER = "Anger"
    FEAR = "Fear"
    SURPRISE = "Surprise"
    HUMOR = "Humor"
    LOVE = "Love"

    @classmethod
    def from_str(cls, emotion: str) -> str:
        return cls[emotion.upper()].value


class EmotionItem(BaseModel):
    """This class is used in EmotionAnalysisDataClass to list emotion analysed.
    Args:
        - emotion (EmotionEnum): emotion of the text
        - emotion_score (float): score of the emotion
    """
    emotion: str
    emotion_score: float = Field(ge=0, le=1)




class EmotionDetectionDataClass(BaseModel):
    """This class is used to standardize the responses from emotion_detection.
    Args:
        - text (str) : The text analysed
        - items (Sequence[EmotionItem]): Lists of the different emotion analysed.

    """
    text: str
    items: Sequence[EmotionItem] = Field(default_factory=list)
