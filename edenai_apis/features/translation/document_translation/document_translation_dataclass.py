from pydantic import BaseModel, StrictStr


class DocumentTranslationDataClass(BaseModel):
    file: str
