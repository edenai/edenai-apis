from pydantic import BaseModel


class AutomlClassificationPredictAsyncDataClass(BaseModel):
    label: str
    confidence: float
