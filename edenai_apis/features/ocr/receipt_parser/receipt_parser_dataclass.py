from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class CustomerInformation(BaseModel):
    customer_name: Optional[StrictStr] = None


class MerchantInformation(BaseModel):
    merchant_name: Optional[StrictStr] = None
    merchant_address: Optional[StrictStr] = None
    merchant_phone: Optional[StrictStr] = None
    merchant_url: Optional[StrictStr] = None
    merchant_siret: Optional[StrictStr] = None
    merchant_siren: Optional[StrictStr] = None
    merchant_vat_number: Optional[StrictStr] = None
    merchant_gst_number: Optional[StrictStr] = None
    merchant_abn_number: Optional[StrictStr] = None


class PaymentInformation(BaseModel):
    card_type: Optional[StrictStr] = None
    card_number: Optional[StrictStr] = None
    cash: Optional[StrictStr] = None
    tip: Optional[StrictStr] = None
    discount: Optional[StrictStr] = None
    change: Optional[StrictStr] = None


class Locale(BaseModel):
    currency: Optional[StrictStr] = None
    language: Optional[StrictStr] = None
    country: Optional[StrictStr] = None


class ItemLines(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    unit_price: Optional[float] = None


class Taxes(BaseModel):
    taxes: Optional[float] = None
    rate: Optional[float] = None


class BarCode(BaseModel):
    value: str
    type: str


class InfosReceiptParserDataClass(BaseModel):
    invoice_number: Optional[StrictStr] = None
    invoice_total: Optional[float] = None
    invoice_subtotal: Optional[float] = None
    barcodes: Sequence[BarCode] = Field(default_factory=list)
    category: Optional[StrictStr] = None
    date: Optional[StrictStr] = None
    due_date: Optional[StrictStr] = None
    time: Optional[StrictStr] = None
    customer_information: CustomerInformation = CustomerInformation()
    merchant_information: MerchantInformation = MerchantInformation()
    payment_information: PaymentInformation = PaymentInformation()
    locale: Locale = Locale()
    taxes: Sequence[Taxes] = Field(default_factory=list)
    receipt_infos: Dict[str, object] = Field(
        default_factory=dict
    )  # DEPRECATED MUST BE DELETED
    item_lines: Sequence[ItemLines] = Field(default_factory=list)


# DEORECATED
class ReceiptParserDataClass(BaseModel):
    extracted_data: Sequence[InfosReceiptParserDataClass] = Field(default_factory=list)
