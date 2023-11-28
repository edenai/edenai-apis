from typing import Optional

from pydantic import BaseModel


class AutomlClassificationCreateProjectDataClass(BaseModel):
    status: str
    name: Optional[str]
    project_id: str
