from pydantic import BaseModel


class QuestionAnswerDataClass(BaseModel):
    answer: str
