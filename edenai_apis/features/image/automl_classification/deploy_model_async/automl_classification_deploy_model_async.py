from pydantic import BaseModel


class AutomlClassificationDeployModel(BaseModel):
    deployed: bool
