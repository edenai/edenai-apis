from pydantic import BaseModel, StrictStr


class AutomlClassificationRemoveImage(BaseModel):
    removed: bool
