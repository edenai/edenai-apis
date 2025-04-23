from typing import Optional
from pydantic import BaseModel, StrictStr


class AutomlClassificationAnnotationDataClass(BaseModel):
    labelId: StrictStr
    label_name: Optional[StrictStr] = None


class AutomlClassificationPredictionDataClass(BaseModel):
    labelId: StrictStr
    confidence: float
    label_name: Optional[StrictStr] = None


class AutomlClassificationListEntryDataClass(BaseModel):
    annotation: AutomlClassificationAnnotationDataClass
    data: StrictStr
    externalId: Optional[StrictStr] = None
    id: StrictStr
    prediction: Optional[AutomlClassificationPredictionDataClass] = None
    final_status: StrictStr


class AutomlClassificationListDataClass(BaseModel):
    items: list[AutomlClassificationListEntryDataClass]


class AutomlCalssificationErrorDataClass(BaseModel):
    message: StrictStr
    status: StrictStr
