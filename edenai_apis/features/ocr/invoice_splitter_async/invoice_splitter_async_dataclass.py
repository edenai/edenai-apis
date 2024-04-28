from typing import List

from pydantic import BaseModel, field_validator


class InvoiceGroupDataClass(BaseModel):
    page_indexes: List[int]
    confidence: float

    @classmethod
    @field_validator("confidence")
    def confidence_validator(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 2)


class InvoiceSplitterAsyncDataClass(BaseModel):
    extracted_data: List[InvoiceGroupDataClass]
