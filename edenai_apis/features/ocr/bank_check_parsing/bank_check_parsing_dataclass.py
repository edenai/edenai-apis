from typing import Optional, Sequence
from pydantic import BaseModel, Field


class MicrModel(BaseModel):
    raw: Optional[str] = Field(...)
    account_number: Optional[str] = Field(...)
    routing_number: Optional[str] = Field(...)
    serial_number: Optional[str] = Field(...)
    check_number: Optional[str] = Field(...)


class ItemBankCheckParsingDataClass(BaseModel):
    amount: Optional[float] = Field(...)
    amount_text: Optional[str] = Field(...)
    bank_address: Optional[str] = Field(...)
    bank_name: Optional[str] = Field(...)
    date: Optional[str] = Field(...)
    memo: Optional[str] = Field(...)
    payer_address: Optional[str] = Field(...)
    payer_name: Optional[str] = Field(...)
    receiver_address: Optional[str] = Field(...)
    receiver_name: Optional[str] = Field(...)
    currency: Optional[str] = Field(...)
    micr: MicrModel = Field(...)


class BankCheckParsingDataClass(BaseModel):
    extracted_data: Sequence[ItemBankCheckParsingDataClass] = Field(
        default_factory=list
    )
