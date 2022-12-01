from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class CustomerInformation(BaseModel):
    customer_name: Optional[StrictStr]



class MerchantInformation(BaseModel):
    merchant_name: Optional[StrictStr]
    merchant_address : Optional[StrictStr]
    merchant_phone: Optional[StrictStr]
    merchant_url: Optional[StrictStr]
    merchant_siret: Optional[StrictStr]
    merchant_siren: Optional[StrictStr]

class PaymentInformation(BaseModel):
    card_type : Optional[StrictStr]
    card_number : Optional[StrictStr]
    cash : Optional[StrictStr]
    tip : Optional[StrictStr]
    discount : Optional[StrictStr]
    change : Optional[StrictStr]

class Locale(BaseModel):
    currency: Optional[StrictStr]
    language: Optional[StrictStr]
    country: Optional[StrictStr]



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
    invoice_subtotal: Optional[float]
    barcodes : Sequence[StrictStr] = Field(default_factory=list)
    category : Optional[StrictStr]
    date: Optional[StrictStr]
    due_date: Optional[StrictStr]
    time : Optional[StrictStr]
    customer_information: CustomerInformation = CustomerInformation()
    merchant_information: MerchantInformation = MerchantInformation()
    payment_information : PaymentInformation = PaymentInformation()
    locale: Locale = Locale()
    taxes: Sequence[Taxes] = Field(default_factory=list)
    receipt_infos: Dict[str, object] = Field(default_factory=dict) # DEPRECATED MUST BE DELETED
    item_lines: Sequence[ItemLines] = Field(default_factory=list)


class ReceiptParserDataClass(BaseModel):
    extracted_data: Sequence[InfosReceiptParserDataClass] = Field(default_factory=list)
