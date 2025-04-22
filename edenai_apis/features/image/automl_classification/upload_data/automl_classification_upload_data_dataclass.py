from typing import Optional
from pydantic import BaseModel, StrictStr


class AutomlClassificationUploadDataDataClass(BaseModel):
    annotation: dict
    data: StrictStr
    externalId: Optional[StrictStr] = None
    id: StrictStr
    prediction: Optional[dict] = None
