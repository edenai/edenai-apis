from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class CustomerInformationInvoice(BaseModel):
    customer_name: Optional[StrictStr]
    customer_address : Optional[StrictStr]  # Deprecated need to be removed
    customer_email : Optional[StrictStr] # New
    customer_id : Optional[StrictStr] # New
    customer_tax_id : Optional[StrictStr] # New
    customer_mailing_address : Optional[StrictStr] # New
    customer_billing_address: Optional[StrictStr] # New
    customer_shipping_address : Optional[StrictStr] # New
    customer_service_address : Optional[StrictStr] # New
    customer_remittance_address : Optional[StrictStr] # New


class MerchantInformationInvoice(BaseModel):
    merchant_name: Optional[StrictStr]
    merchant_address: Optional[StrictStr]
    merchant_phone : Optional[StrictStr] # New
    merchant_email : Optional[StrictStr] # New
    merchant_fax : Optional[StrictStr] # New
    merchant_website : Optional[StrictStr] # New
    merchant_tax_id : Optional[StrictStr] # New
    merchant_siret : Optional[StrictStr] # New
    merchant_siren : Optional[StrictStr] # New
    


class LocaleInvoice(BaseModel):
    currency: Optional[StrictStr]
    language: Optional[StrictStr]


class ItemLinesInvoice(BaseModel):
    description: Optional[StrictStr]
    quantity: Optional[int]
    amount: Optional[float]
    unit_price: Optional[float]
    discount : Optional[int] # New
    product_code : Optional[StrictStr] # New
    date_item : Optional[StrictStr] # New
    tax_item : Optional[float] # New


class TaxesInvoice(BaseModel):
    value: Optional[float]
    rate: Optional[float] 

class BankInvoice(BaseModel): # New obj
    account_number : Optional[StrictStr] # New
    iban : Optional[StrictStr] # New
    bsb : Optional[StrictStr] # New
    sort_code : Optional[StrictStr] # New
    vat_number : Optional[StrictStr] # New
    rooting_number : Optional[StrictStr] # New
    swift: Optional[StrictStr]
    
class InfosInvoiceParserDataClass(BaseModel):
    customer_information: CustomerInformationInvoice = CustomerInformationInvoice()
    merchant_information: MerchantInformationInvoice = MerchantInformationInvoice()
    #--------------------------------------------#
    invoice_number: Optional[StrictStr]
    invoice_total: Optional[float]
    invoice_subtotal: Optional[float]
    amount_due : Optional[float] # New
    previous_unpaid_balance : Optional[float] # New
    discount : Optional[float] # New
    taxes: Sequence[TaxesInvoice] = Field(default_factory=list) # Change from list to item -> total_tax
    #--------------------------------------------#
    payment_term : Optional[StrictStr] # New
    purchase_order : Optional[StrictStr] # New
    date: Optional[StrictStr]
    due_date : Optional[StrictStr]
    service_date : Optional[StrictStr] # New
    service_due_date : Optional[StrictStr] # New
    #--------------------------------------------#
    locale: LocaleInvoice = LocaleInvoice()
    bank_informations : BankInvoice = BankInvoice() # New
    #--------------------------------------------#
    item_lines: Sequence[ItemLinesInvoice] = Field(default_factory=list)

class InvoiceParserDataClass(BaseModel):
    extracted_data: Sequence[InfosInvoiceParserDataClass] = Field(default_factory=list)
