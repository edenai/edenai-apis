from typing import Optional, List

from pydantic import BaseModel, Field, StrictStr
from enum import Enum


class FinancialParserType(Enum):
    RECEIPT = "receipt"
    INVOICE = "invoice"


class FinancialCustomerInformation(BaseModel):
    name: Optional[StrictStr] = Field(
        default=None, description="The name of the invoiced customer."
    )
    id_reference: Optional[StrictStr] = Field(
        default=None, description="Unique reference ID for the customer."
    )
    mailling_address: Optional[StrictStr] = Field(
        default=None, description="The mailing address of the customer."
    )
    billing_address: Optional[StrictStr] = Field(
        default=None, description="The explicit billing address for the customer."
    )
    shipping_address: Optional[StrictStr] = Field(
        default=None, description="The shipping address for the customer."
    )
    service_address: Optional[StrictStr] = Field(
        default=None, description="The service address associated with the customer."
    )
    remittance_address: Optional[StrictStr] = Field(
        default=None, description="The address to which payments should be remitted."
    )
    email: Optional[StrictStr] = Field(
        default=None, description="The email address of the customer."
    )
    phone: Optional[StrictStr] = Field(
        default=None, description="The phone number associated with the customer."
    )
    vat_number: Optional[StrictStr] = Field(
        default=None, description="VAT (Value Added Tax) number of the customer."
    )
    abn_number: Optional[StrictStr] = Field(
        default=None, description="ABN (Australian Business Number) of the customer."
    )
    gst_number: Optional[StrictStr] = Field(
        default=None, description="GST (Goods and Services Tax) number of the customer."
    )
    pan_number: Optional[StrictStr] = Field(
        default=None, description="PAN (Permanent Account Number) of the customer."
    )
    business_number: Optional[StrictStr] = Field(
        default=None, description="Business registration number of the customer."
    )
    siret_number: Optional[StrictStr] = Field(
        default=None,
        description="SIRET (Système d'Identification du Répertoire des Entreprises et de leurs Établissements) number of the customer.",
    )
    siren_number: Optional[StrictStr] = Field(
        default=None,
        description="SIREN (Système d'Identification du Répertoire des Entreprises) number of the customer.",
    )
    customer_number: Optional[StrictStr] = Field(
        default=None, description="Customer identification number."
    )
    coc_number: Optional[StrictStr] = Field(
        default=None, description="Chamber of Commerce registration number."
    )
    fiscal_number: Optional[StrictStr] = Field(
        default=None, description="Fiscal identification number of the customer."
    )
    registration_number: Optional[StrictStr] = Field(
        default=None, description="Official registration number of the customer."
    )
    tax_id: Optional[StrictStr] = Field(
        default=None, description="Tax identification number of the customer."
    )
    website: Optional[StrictStr] = Field(
        default=None, description="The website associated with the customer."
    )
    remit_to_name: Optional[StrictStr] = Field(
        default=None,
        description="The name associated with the customer's remittance address.",
    )
    city: Optional[StrictStr] = Field(
        default=None, description="The city associated with the customer's address."
    )
    country: Optional[StrictStr] = Field(
        default=None, description="The country associated with the customer's address."
    )
    house_number: Optional[StrictStr] = Field(
        default=None,
        description="The house number associated with the customer's address.",
    )
    province: Optional[StrictStr] = Field(
        default=None, description="The province associated with the customer's address."
    )
    street_name: Optional[StrictStr] = Field(
        default=None,
        description="The street name associated with the customer's address.",
    )
    zip_code: Optional[StrictStr] = Field(
        default=None, description="The ZIP code associated with the customer's address."
    )
    municipality: Optional[StrictStr] = Field(
        default=None,
        description="The municipality associated with the customer's address.",
    )


class FinancialMerchantInformation(BaseModel):
    name: Optional[StrictStr] = Field(default=None, description="Name of the merchant.")
    address: Optional[StrictStr] = Field(
        default=None, description="Address of the merchant."
    )
    phone: Optional[StrictStr] = Field(
        default=None, description="Phone number of the merchant."
    )
    tax_id: Optional[StrictStr] = Field(
        default=None, description="Tax identification number of the merchant."
    )
    id_reference: Optional[StrictStr] = Field(
        default=None, description="Unique reference ID for the merchant."
    )
    vat_number: Optional[StrictStr] = Field(
        default=None, description="VAT (Value Added Tax) number of the merchant."
    )
    abn_number: Optional[StrictStr] = Field(
        default=None, description="ABN (Australian Business Number) of the merchant."
    )
    gst_number: Optional[StrictStr] = Field(
        default=None, description="GST (Goods and Services Tax) number of the merchant."
    )
    business_number: Optional[StrictStr] = Field(
        default=None, description="Business registration number of the merchant."
    )
    siret_number: Optional[StrictStr] = Field(
        default=None,
        description="SIRET (Système d'Identification du Répertoire des Entreprises et de leurs Établissements) number of the merchant.",
    )
    siren_number: Optional[StrictStr] = Field(
        default=None,
        description="SIREN (Système d'Identification du Répertoire des Entreprises) number of the merchant.",
    )
    pan_number: Optional[StrictStr] = Field(
        default=None, description="PAN (Permanent Account Number) of the merchant."
    )
    coc_number: Optional[StrictStr] = Field(
        default=None,
        description="Chamber of Commerce registration number of the merchant.",
    )
    fiscal_number: Optional[StrictStr] = Field(
        default=None, description="Fiscal identification number of the merchant."
    )
    email: Optional[StrictStr] = Field(
        default=None, description="Email address of the merchant."
    )
    fax: Optional[StrictStr] = Field(
        default=None, description="Fax number of the merchant."
    )
    website: Optional[StrictStr] = Field(
        default=None, description="Website of the merchant."
    )
    registration: Optional[StrictStr] = Field(
        default=None, description="Official registration information of the merchant."
    )
    city: Optional[StrictStr] = Field(
        default=None, description="City associated with the merchant's address."
    )
    country: Optional[StrictStr] = Field(
        default=None, description="Country associated with the merchant's address."
    )
    house_number: Optional[StrictStr] = Field(
        default=None, description="House number associated with the merchant's address."
    )
    province: Optional[StrictStr] = Field(
        default=None, description="Province associated with the merchant's address."
    )
    street_name: Optional[StrictStr] = Field(
        default=None, description="Street name associated with the merchant's address."
    )
    zip_code: Optional[StrictStr] = Field(
        default=None, description="ZIP code associated with the merchant's address."
    )
    country_code: Optional[StrictStr] = Field(
        default=None,
        description="Country code associated with the merchant's location.",
    )


class FinancialLocalInformation(BaseModel):
    currency: Optional[StrictStr] = Field(
        default=None, description="Currency used in financial transactions."
    )
    currency_code: Optional[StrictStr] = Field(
        default=None, description="Currency code (e.g., USD, EUR)."
    )
    currency_exchange_rate: Optional[StrictStr] = Field(
        default=None, description="Exchange rate for the specified currency."
    )
    country: Optional[StrictStr] = Field(
        default=None,
        description="Country associated with the local financial information.",
    )
    language: Optional[StrictStr] = Field(
        default=None, description="Language used in financial transactions."
    )


class FinancialPaymentInformation(BaseModel):
    amount_due: Optional[float] = Field(
        default=None, description="Amount due for payment."
    )
    amount_tip: Optional[float] = Field(
        default=None, description="Tip amount in a financial transaction."
    )
    amount_shipping: Optional[float] = Field(
        default=None, description="Shipping cost in a financial transaction."
    )
    amount_change: Optional[float] = Field(
        default=None, description="Change amount in a financial transaction."
    )
    amount_paid: Optional[float] = Field(
        default=None, description="Amount already paid in a financial transaction."
    )
    total: Optional[float] = Field(
        default=None, description="Total amount in the invoice."
    )
    subtotal: Optional[float] = Field(
        default=None, description="Subtotal amount in a financial transaction."
    )
    total_tax: Optional[float] = Field(
        default=None, description="Total tax amount in a financial transaction."
    )
    tax_rate: Optional[float] = Field(
        default=None, description="Tax rate applied in a financial transaction."
    )
    discount: Optional[float] = Field(
        default=None, description="Discount amount applied in a financial transaction."
    )
    gratuity: Optional[float] = Field(
        default=None, description="Gratuity amount in a financial transaction."
    )
    service_charge: Optional[float] = Field(
        default=None, description="Service charge in a financial transaction."
    )
    previous_unpaid_balance: Optional[float] = Field(
        default=None, description="Previous unpaid balance in a financial transaction."
    )
    prior_balance: Optional[float] = Field(
        default=None,
        description="Prior balance before the current financial transaction.",
    )
    payment_terms: Optional[StrictStr] = Field(
        default=None, description="Terms and conditions for payment."
    )
    payment_method: Optional[StrictStr] = Field(
        default=None, description="Payment method used in the financial transaction."
    )
    payment_card_number: Optional[StrictStr] = Field(
        default=None, description="Card number used in the payment."
    )
    payment_auth_code: Optional[StrictStr] = Field(
        default=None, description="Authorization code for the payment."
    )
    shipping_handling_charge: Optional[float] = Field(
        default=None,
        description="Charge for shipping and handling in a financial transaction.",
    )
    transaction_number: Optional[StrictStr] = Field(
        default=None, description="Unique identifier for the financial transaction."
    )
    transaction_reference: Optional[StrictStr] = Field(
        default=None, description="Reference number for the financial transaction."
    )


class FinancialBankInformation(BaseModel):
    iban: Optional[StrictStr] = Field(
        default=None, description="International Bank Account Number."
    )
    swift: Optional[StrictStr] = Field(
        default=None,
        description="Society for Worldwide Interbank Financial Telecommunication code.",
    )
    bsb: Optional[StrictStr] = Field(
        default=None, description="Bank State Branch code (Australia)."
    )
    sort_code: Optional[StrictStr] = Field(
        default=None, description="Sort code for UK banks."
    )
    account_number: Optional[StrictStr] = Field(
        default=None, description="Bank account number."
    )
    routing_number: Optional[StrictStr] = Field(
        default=None, description="Routing number for banks in the United States."
    )
    bic: Optional[StrictStr] = Field(default=None, description="Bank Identifier Code.")


class FinancialLineItem(BaseModel):
    tax: Optional[float] = Field(
        default=None, description="Tax amount for the line item."
    )
    amount_line: Optional[float] = Field(
        default=None, description="Total amount for the line item."
    )
    description: Optional[StrictStr] = Field(
        default=None, description="Description of the line item."
    )
    quantity: Optional[float] = Field(
        default=None, description="Quantity of units for the line item."
    )
    unit_price: Optional[float] = Field(
        default=None, description="Unit price for each unit in the line item."
    )
    unit_type: Optional[StrictStr] = Field(
        default=None, description="Type of unit (e.g., hours, items)."
    )
    date: Optional[StrictStr] = Field(
        default=None, description="Date associated with the line item."
    )
    product_code: Optional[StrictStr] = Field(
        default=None, description="Product code or identifier for the line item."
    )
    purchase_order: Optional[StrictStr] = Field(
        default=None, description="Purchase order related to the line item."
    )
    tax_rate: Optional[float] = Field(
        default=None, description="Tax rate applied to the line item."
    )
    base_total: Optional[float] = Field(
        default=None, description="Base total amount before any discounts or taxes."
    )
    sub_total: Optional[float] = Field(
        default=None, description="Subtotal amount for the line item."
    )
    discount_amount: Optional[float] = Field(
        default=None, description="Amount of discount applied to the line item."
    )
    discount_rate: Optional[float] = Field(
        default=None, description="Rate of discount applied to the line item."
    )
    discount_code: Optional[StrictStr] = Field(
        default=None,
        description="Code associated with any discount applied to the line item.",
    )
    order_number: Optional[StrictStr] = Field(
        default=None, description="Order number associated with the line item."
    )
    title: Optional[StrictStr] = Field(
        default=None, description="Title or name of the line item."
    )


class FinancialBarcode(BaseModel):
    value: str
    type: str


class FinancialDocumentInformation(BaseModel):
    invoice_receipt_id: Optional[StrictStr] = Field(
        default=None, description="Identifier for the invoice."
    )
    purchase_order: Optional[StrictStr] = Field(
        default=None, description="Purchase order related to the document."
    )
    invoice_date: Optional[StrictStr] = Field(
        default=None, description="Date of the invoice."
    )
    time: Optional[StrictStr] = Field(
        default=None, description="Time associated with the document."
    )
    invoice_due_date: Optional[StrictStr] = Field(
        default=None, description="Due date for the invoice."
    )
    service_start_date: Optional[StrictStr] = Field(
        default=None,
        description="Start date of the service associated with the document.",
    )
    service_end_date: Optional[StrictStr] = Field(
        default=None,
        description="End date of the service associated with the document.",
    )
    reference: Optional[StrictStr] = Field(
        default=None, description="Reference number associated with the document."
    )
    biller_code: Optional[StrictStr] = Field(
        default=None, description="Biller code associated with the document."
    )
    order_date: Optional[StrictStr] = Field(
        default=None, description="Date of the order associated with the document."
    )
    tracking_number: Optional[StrictStr] = Field(
        default=None, description="Tracking number associated with the document."
    )
    barcodes: List[FinancialBarcode] = Field(
        default_factory=list,
        description="List of barcodes associated with the document.",
    )


class FinancialDocumentMetadata(BaseModel):
    document_index: Optional[int] = Field(
        default=None, description="Index of the detected document."
    )
    document_page_number: Optional[int] = Field(
        default=None, description="Page number within the document."
    )
    document_type: Optional[StrictStr] = Field(
        default=None, description="Type or category of the document."
    )


class FinancialParserObjectDataClass(BaseModel):
    customer_information: FinancialCustomerInformation
    merchant_information: FinancialMerchantInformation
    payment_information: FinancialPaymentInformation
    financial_document_information: FinancialDocumentInformation
    local: FinancialLocalInformation
    bank: FinancialBankInformation
    item_lines: List[FinancialLineItem] = Field(
        default_factory=list,
        description="List of line items associated with the document.",
    )
    document_metadata: FinancialDocumentMetadata


class FinancialParserDataClass(BaseModel):
    extracted_data: List[FinancialParserObjectDataClass] = Field(
        default_factory=list,
        description="List of parsed financial data objects (per page).",
    )
