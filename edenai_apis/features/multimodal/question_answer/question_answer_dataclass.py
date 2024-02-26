from typing import List

from pydantic import Field, BaseModel


class QuestionAnswerDataClass(BaseModel):
    answers: List[str] = Field(default_factory=list)
