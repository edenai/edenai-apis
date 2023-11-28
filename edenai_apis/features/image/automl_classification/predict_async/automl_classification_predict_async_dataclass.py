from pydantic import BaseModel


class AutomlClassificationPredictDataClass(BaseModel):
    label: str
    confidence: float
