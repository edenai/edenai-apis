from typing import Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class CustomerInformationInvoice(BaseModel):
    customer_name: Optional[StrictStr]
    customer_address: Optional[StrictStr]


class MerchantInformationInvoice(BaseModel):
    merchant_name: Optional[StrictStr]
    merchant_address: Optional[StrictStr]


class LocaleInvoice(BaseModel):
    currency: Optional[StrictStr]
    language: Optional[StrictStr]


class ItemLinesInvoice(BaseModel):
    description: StrictStr
    quantity: int
    amount: float
    unit_price: float


class TaxesInvoice(BaseModel):
    value: Optional[float]
    rate: Optional[float]

class InfosInvoiceParserDataClass(BaseModel):
    invoice_number: Optional[StrictStr]
    invoice_total: Optional[float]
    date: Optional[StrictStr]
    invoice_subtotal: Optional[float]
    due_date: Optional[StrictStr]
    customer_information: CustomerInformationInvoice = CustomerInformationInvoice()
    merchant_information: MerchantInformationInvoice = MerchantInformationInvoice()
    locale: LocaleInvoice = LocaleInvoice()
    taxes: Sequence[TaxesInvoice] = Field(default_factory=list)
    item_lines: Sequence[ItemLinesInvoice] = Field(default_factory=list)

class InvoiceParserDataClass(BaseModel):
    extracted_data: Sequence[InfosInvoiceParserDataClass] = Field(default_factory=list)
