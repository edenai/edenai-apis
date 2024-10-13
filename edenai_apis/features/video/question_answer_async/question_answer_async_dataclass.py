from pydantic import BaseModel


class QuestionAnswerAsyncDataClass(BaseModel):
    answer: str
