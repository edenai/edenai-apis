from typing import Sequence
from pydantic import BaseModel, Field

class EmotionDataClass(BaseModel):
    """This class is used in EmotionAnalysisDataClass to list emotion analysed.
    Args:
        - emotion (str): emotion of the text
        - emotion_score (float): score of the emotion
    """
    emotion: str
    emotion_score: float

class EmotionDetectionDataClass(BaseModel):
    """This class is used to standardize the responses from emotion_detection.
    Args:
        - text (str) : The text analysed
        - items (Sequence[EmotionDataClass]): Lists of the different emotion analysed.

    """
    text: str
    items: Sequence[EmotionDataClass] = Field(default_factory=list)