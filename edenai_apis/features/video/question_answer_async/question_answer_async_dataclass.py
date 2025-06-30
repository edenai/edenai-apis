from typing import Optional
from pydantic import BaseModel


class QuestionAnswerAsyncDataClass(BaseModel):
    answer: str
    finish_reason: Optional[str]
