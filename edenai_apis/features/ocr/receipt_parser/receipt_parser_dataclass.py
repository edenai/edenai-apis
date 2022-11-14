from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class CustomerInformation(BaseModel):
    customer_name: Optional[StrictStr]



class MerchantInformation(BaseModel):
    merchant_name: Optional[StrictStr]



class Locale(BaseModel):
    currency: Optional[StrictStr]
    # language: Optional[StrictStr]



class ItemLines(BaseModel):
    description: Optional[StrictStr]
    quantity: int
    amount: Optional[float]
    unit_price: Optional[float]



class Taxes(BaseModel):
    taxes: Optional[float]
    rate: Optional[float]


class InfosReceiptParserDataClass(BaseModel):
    invoice_number: Optional[StrictStr]
    invoice_total: Optional[float]
    date: Optional[StrictStr]
    invoice_subtotal: Optional[float]
    customer_information: CustomerInformation = CustomerInformation()
    merchant_information: MerchantInformation = MerchantInformation()
    locale: Locale = Locale()
    taxes: Sequence[Taxes] = Field(default_factory=list)
    receipt_infos: Dict[str, object] = Field(default_factory=dict)
    item_lines: Sequence[ItemLines] = Field(default_factory=list)


class ReceiptParserDataClass(BaseModel):
    extracted_data: Sequence[InfosReceiptParserDataClass] = Field(default_factory=list)
