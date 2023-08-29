from pydantic import BaseModel, StrictStr


class AutomlClassificationPrediction(BaseModel):
    label: StrictStr
    confidence: float
