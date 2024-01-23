from pydantic import BaseModel


class GenerationFineTuningCreateProjectAsyncDataClass(BaseModel):
    project_id: str
    name: str
    description: str
