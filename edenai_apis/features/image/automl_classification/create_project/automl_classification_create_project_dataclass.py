from typing import Optional

from pydantic import BaseModel


class AutomlClassificationCreateProjectDataClass(BaseModel):
    name: Optional[str]
    project_id: str
