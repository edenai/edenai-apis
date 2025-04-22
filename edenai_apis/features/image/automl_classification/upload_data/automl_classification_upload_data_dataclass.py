from pydantic import BaseModel, StrictStr


class AutomlClassificationUploadDataDataClass(BaseModel):
    message: str
    image: StrictStr
    label_name: str
