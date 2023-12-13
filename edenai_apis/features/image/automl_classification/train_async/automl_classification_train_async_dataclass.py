from typing import Optional

from pydantic import BaseModel


class AutomlClassificationTrainAsyncDataClass(BaseModel):
    message: str
    name: Optional[str]
    project_id: str
