from pydantic import BaseModel


class AutomlClassificationUploadDataAsyncDataClass(BaseModel):
    message: str
    label_name: str
