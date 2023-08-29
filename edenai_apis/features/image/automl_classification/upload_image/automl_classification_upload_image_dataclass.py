from pydantic import BaseModel, StrictStr


class AutomlClassificationUploadImage(BaseModel):
    image_id: StrictStr
