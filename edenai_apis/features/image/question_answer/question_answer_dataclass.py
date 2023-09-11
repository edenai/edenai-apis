from typing import Sequence
from pydantic import BaseModel, Field


class QuestionAnswerDataClass(BaseModel):
    answers: Sequence[str] = Field(default_factory=list)
