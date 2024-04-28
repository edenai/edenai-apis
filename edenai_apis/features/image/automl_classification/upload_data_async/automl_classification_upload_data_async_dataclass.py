from pydantic import BaseModel, StrictStr


class AutomlClassificationUploadDataAsyncDataClass(BaseModel):
    message: str
    image: StrictStr
    label_name: str
