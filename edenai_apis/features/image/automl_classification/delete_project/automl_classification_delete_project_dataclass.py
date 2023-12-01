from pydantic import BaseModel


class AutomlClassificationDeleteProjectDataClass(BaseModel):
    deleted: bool
