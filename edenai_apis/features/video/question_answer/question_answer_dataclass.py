from typing import Optional
from pydantic import BaseModel


class QuestionAnswerDataClass(BaseModel):
    answer: str
    finish_reason: Optional[str]
