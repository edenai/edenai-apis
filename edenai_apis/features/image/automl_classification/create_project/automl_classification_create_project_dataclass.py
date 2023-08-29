from pydantic import BaseModel, StrictStr


class AutomlClassificationCreateProject(BaseModel):
    project_id: StrictStr
