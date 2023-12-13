from typing import Optional, List

from pydantic import BaseModel, Field, StrictStr
from enum import Enum

class FinancialParserType(Enum):
    RECEIPT = "receipt"
    INVOICE = "invoice"
class FinancialCustomerInformation(BaseModel):
    name : Optional[StrictStr] = Field(default=None)
    id_reference: Optional[StrictStr] = Field(default=None)
    mailling_address: Optional[StrictStr] = Field(default=None)
    billing_address: Optional[StrictStr] = Field(default=None)
    shipping_address: Optional[StrictStr] = Field(default=None)
    service_address: Optional[StrictStr] = Field(default=None)
    remittance_address: Optional[StrictStr] = Field(default=None)
    email: Optional[StrictStr] = Field(default=None)
    phone: Optional[StrictStr] = Field(default=None) 
    vat_number: Optional[StrictStr] = Field(default=None)
    abn_number: Optional[StrictStr] = Field(default=None)
    gst_number: Optional[StrictStr] = Field(default=None)
    pan_number: Optional[StrictStr] = Field(default=None)
    business_number: Optional[StrictStr] = Field(default=None)
    siret_number: Optional[StrictStr] = Field(default=None)
    siren_number: Optional[StrictStr] = Field(default=None)
    customer_number: Optional[StrictStr] = Field(default=None)
    coc_number: Optional[StrictStr] = Field(default=None)
    fiscal_number: Optional[StrictStr] = Field(default=None)
    registration_number: Optional[StrictStr] = Field(default=None)
    tax_id: Optional[StrictStr] = Field(default=None)
    website: Optional[StrictStr] = Field(default=None)
    remit_to_name: Optional[StrictStr] = Field(default=None)
    city: Optional[StrictStr] = Field(default=None)
    country: Optional[StrictStr] = Field(default=None)
    house_number: Optional[StrictStr] = Field(default=None)
    province: Optional[StrictStr] = Field(default=None)
    street_name: Optional[StrictStr] = Field(default=None)
    zip_code: Optional[StrictStr] = Field(default=None)
    municipality: Optional[StrictStr] = Field(default=None)

class FinancialMerchantInformation(BaseModel):
    name: Optional[StrictStr] = Field(default=None)
    address: Optional[StrictStr] = Field(default=None)
    phone: Optional[StrictStr] = Field(default=None)
    tax_id: Optional[StrictStr] = Field(default=None)
    id_reference: Optional[StrictStr] = Field(default=None)
    vat_number: Optional[StrictStr] = Field(default=None)
    abn_number: Optional[StrictStr] = Field(default=None)
    gst_number: Optional[StrictStr] = Field(default=None)
    business_number: Optional[StrictStr] = Field(default=None)
    siret_number: Optional[StrictStr] = Field(default=None)
    siren_number: Optional[StrictStr] = Field(default=None)
    pan_number: Optional[StrictStr] = Field(default=None)
    coc_number: Optional[StrictStr] = Field(default=None)
    fiscal_number: Optional[StrictStr] = Field(default=None)
    email: Optional[StrictStr] = Field(default=None) 
    fax: Optional[StrictStr] = Field(default=None)
    website: Optional[StrictStr] = Field(default=None)
    registration: Optional[StrictStr] = Field(default=None)
    city: Optional[StrictStr] = Field(default=None)
    country: Optional[StrictStr] = Field(default=None)
    house_number: Optional[StrictStr] = Field(default=None)
    province: Optional[StrictStr] = Field(default=None)
    street_name: Optional[StrictStr] = Field(default=None)
    zip_code: Optional[StrictStr] = Field(default=None)
    country_code: Optional[StrictStr] = Field(default=None)
    
class FinancialLocalInformation(BaseModel):
    currency: Optional[StrictStr] = Field(default=None)
    currency_code: Optional[StrictStr] = Field(default=None)
    currency_exchange_rate: Optional[StrictStr] = Field(default=None)
    country: Optional[StrictStr] = Field(default=None)
    language: Optional[StrictStr] = Field(default=None)

class FinancialPaymentInformation(BaseModel):
    amount_due: Optional[float] = Field(default=None)
    amount_tip: Optional[float] = Field(default=None)
    amount_shipping: Optional[float] = Field(default=None)
    amount_change: Optional[float] = Field(default=None)
    amount_paid: Optional[float] = Field(default=None)
    invoice_total: Optional[float] = Field(default=None)
    subtotal: Optional[float] = Field(default=None)
    total_tax: Optional[float] = Field(default=None)
    tax_rate: Optional[float] = Field(default=None)
    discount: Optional[float] = Field(default=None)
    gratuity: Optional[float] = Field(default=None)
    service_charge: Optional[float] = Field(default=None)
    previous_unpaid_balance: Optional[float] = Field(default=None)
    prior_balance: Optional[float] = Field(default=None)
    payment_terms: Optional[StrictStr] = Field(default=None)
    payment_method: Optional[StrictStr] = Field(default=None)
    payment_card_number: Optional[StrictStr] = Field(default=None)
    payment_auth_code: Optional[StrictStr] = Field(default=None)
    shipping_handling_charge: Optional[float] = Field(default=None)
    transaction_number: Optional[StrictStr] = Field(default=None)
    transaction_reference: Optional[StrictStr] = Field(default=None)

class FinancialBankInformation(BaseModel):
    iban: Optional[StrictStr] = Field(default=None)
    swift: Optional[StrictStr] = Field(default=None)
    bsb: Optional[StrictStr] = Field(default=None)
    sort_code: Optional[StrictStr] = Field(default=None)
    account_number: Optional[StrictStr] = Field(default=None)
    routing_number: Optional[StrictStr] = Field(default=None)
    bic: Optional[StrictStr] = Field(default=None)

class FinancialLineItem(BaseModel):
    tax: Optional[float] = Field(default=None)
    amount_line: Optional[float] = Field(default=None)
    description: Optional[StrictStr] = Field(default=None)
    quantity: Optional[float] = Field(default=None)
    unit_price: Optional[float] = Field(default=None)
    unit_type: Optional[StrictStr] = Field(default=None)
    date: Optional[StrictStr] = Field(default=None)
    product_code: Optional[StrictStr] = Field(default=None)
    purchase_order: Optional[StrictStr] = Field(default=None)
    tax_rate: Optional[float] = Field(default=None)
    base_total: Optional[float] = Field(default=None)
    sub_total: Optional[float] = Field(default=None)
    discount_amount: Optional[float] = Field(default=None)
    discount_rate: Optional[float] = Field(default=None)
    discount_code: Optional[StrictStr] = Field(default=None)
    order_number: Optional[StrictStr] = Field(default=None)
    title: Optional[StrictStr] = Field(default=None)

class FinancialBarcode(BaseModel):
    value: str
    type: str

class FinancialDocumentInformation(BaseModel):
    invoice_id: Optional[StrictStr] = Field(default=None)
    purchase_order: Optional[StrictStr] = Field(default=None)
    invoice_date: Optional[StrictStr] = Field(default=None)
    time: Optional[StrictStr] = Field(default=None)
    invoice_due_date: Optional[StrictStr] = Field(default=None)
    service_start_date: Optional[StrictStr] = Field(default=None)
    service_end_date: Optional[StrictStr] = Field(default=None)
    reference: Optional[StrictStr] = Field(default=None)
    biller_code: Optional[StrictStr] = Field(default=None)
    order_date: Optional[StrictStr] = Field(default=None)
    tracking_number: Optional[StrictStr] = Field(default=None)
    barcodes: List[FinancialBarcode] = Field(default_factory=list)

class FinancialDocumentMetadata(BaseModel):
    document_index: Optional[int] = Field(default=None)
    document_page_number: Optional[int] = Field(default=None)
    document_type: Optional[StrictStr] = Field(default=None)

class FinancialParserObjectDataClass(BaseModel):
    customer_information: FinancialCustomerInformation
    merchant_information: FinancialMerchantInformation 
    payment_information: FinancialPaymentInformation
    financial_document_information: FinancialDocumentInformation
    local: FinancialLocalInformation
    bank: FinancialBankInformation
    item_lines: List[FinancialLineItem] = Field(default_factory=list)
    invoice_metadata: FinancialDocumentMetadata

class FinancialParserDataClass(BaseModel):
    extracted_data: List[FinancialParserObjectDataClass] = Field(default_factory=list)