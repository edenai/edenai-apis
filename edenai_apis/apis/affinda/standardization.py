from locale import currency
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google_crc32c.python import value
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import IdentityParserDataClass, InfoCountry, InfosIdentityParserDataClass, ItemIdentityParserDataClass, format_date, get_info_country
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import BankInvoice, InfosInvoiceParserDataClass, InvoiceParserDataClass, ItemLinesInvoice, TaxesInvoice
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import InfosReceiptParserDataClass, ItemLines, Locale, MerchantInformation, PaymentInformation, ReceiptParserDataClass, Taxes

from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeLocation,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeExtractedData,
    ResumeLang,
    ResumeParserDataClass,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry
)
from edenai_apis.features.ocr.invoice_parser import (
    MerchantInformationInvoice,
    CustomerInformationInvoice
)
from edenai_apis.utils.conversion import combine_date_with_time, convert_string_to_number
from .models import Document, DocumentError, DocumentMeta

class ResumeStandardizer:
    __document: Document
    __data: Dict[str, Any]
    __meta: DocumentMeta
    __error: DocumentError

    __std_response: Dict[str, Any]

    def __init__(self, document: Document) -> None:
        self.__document = document
        self.__data = self.__document.data
        self.__error = self.__document.error
        self.__meta = self.__document.meta
        self.__std_response = {}

    def __std_names(self) -> ResumePersonalName:
        name = self.__data.get("name") or {}
        return ResumePersonalName(
            raw_name=name.get("raw"),
            first_name=name.get("first"),
            last_name=name.get("last"),
            middle=name.get("middle"),
            title=name.get("title"),
            sufix=None,
            prefix=None,
        )

    def __std_location(self, which_location: Optional[dict] = None) -> ResumeLocation: 
        if which_location is None:
            which_location = self.__data
        location = which_location.get("location") or {}
        return ResumeLocation(
            raw_input_location=location.get("rawInput"),
            postal_code=location.get("postalCode"),
            region=location.get("state"),
            country_code=location.get("countryCode"),
            country=location.get("country"),
            appartment_number=location.get("apartmentNumber"),
            city=location.get("city"),
            street=location.get("street"),
            street_number=location.get("streetNumber"),
            formatted_location=location.get("formatted"),
        )

    def std_personnal_information(self) -> ResumePersonalInfo:
        self.__std_response['personal_infos'] = ResumePersonalInfo(
            name=self.__std_names(),
            address=self.__std_location(),
            phones=self.__data.get("phoneNumbers") or [],
            mails=self.__data.get("emails") or [],
            urls=self.__data.get("websites") or [],
            self_summary=self.__data.get("summary"),
            current_profession=self.__data.get("profession"),
            objective=self.__data.get("objective"),
            date_of_birth=self.__data.get("dateOfBirth"),
            place_of_birth=None,
            gender=None,
            nationality=None,
            martial_status=None,
            current_salary=None,
        )

        return self.__std_response['personal_infos']

    def std_education(self) -> ResumeEducation:
        edu_entries: List[ResumeEducationEntry] = []
        for i in (self.__data.get("education") or []):
            dates = i.get("dates") or {}
            grade = i.get("grade") or {}
            edu_entries.append(
                ResumeEducationEntry(
                    location=self.__std_location(which_location=i),
                    start_date=dates.get("startDate"),
                    end_date=dates.get("completionDate"),
                    establishment=i.get("organization"),
                    gpa=grade.get("value"),
                    accreditation=i.get("accreditation", {}).get("education"),
                    title=None,
                    description=None,
                )
            )

        self.__std_response['education'] = ResumeEducation(
            entries=edu_entries,
            total_years_education=None
        )

        return self.__std_response['education']

    def std_work_experience(self) -> ResumeWorkExp:
        work_entries: List[ResumeWorkExpEntry] = []
        for i in (self.__data.get("workExperience") or []):
            dates = i.get("dates") or {}
            work_entries.append(
                ResumeWorkExpEntry(
                    title=i.get("jobTitle"),
                    company=i.get("organization"),
                    start_date=dates.get("startDate"),
                    end_date=dates.get("endDate"),
                    description=i.get("job_description"),
                    location=self.__std_location(which_location=i),
                    industry=None,
                )
            )
        self.__std_response['work_experience'] = ResumeWorkExp(
            total_years_experience=self.__data.get("total_years_experience"),
            entries=work_entries
        )

        return self.__std_response['work_experience']

    def std_skills(self) -> List[ResumeSkill]:
        self.__std_response['skills'] = []
        for i in (self.__data.get("skills", []) or []):
            skill_name = i.get("name")
            skill_type = i.get("type").replace('_skill', '')
            self.__std_response['skills'].append(ResumeSkill(name=skill_name, type=skill_type))

        return self.__std_response['skills']

    def std_miscellaneous(self) -> Tuple[List[ResumeLang], List[ResumeSkill], List[ResumeSkill]]:
        self.__std_response['languages'] = [ResumeLang(name=i, code=None) for i in self.__data.get("languages", [])]
        self.__std_response['certifications'] = [ResumeSkill(name=i, type=None) for i in self.__data.get("certifications", [])]
        self.__std_response['publications'] = [ResumeSkill(name=i, type=None) for i in self.__data.get("publications", [])]
    
        return (
            self.__std_response['languages'],
            self.__std_response['certifications'],
            self.__std_response['publications']
        )

    @property
    def standardized_response(self):
        return ResumeParserDataClass(
            extracted_data=ResumeExtractedData(**self.__std_response)
        )



class InvoiceStandardizer:
    __document: Document
    __data: Dict[str, Any]
    __meta: DocumentMeta
    __error: DocumentError

    __std_response: Dict[str, Any]

    def __init__(self, document: Document) -> None:
        self.__document = document
        self.__data = self.__document.data
        self.__error = self.__document.error
        self.__meta = self.__document.meta
        self.__std_response = {}

    def std_merchant_informations(self) -> MerchantInformationInvoice:
        name = self.__data.get('supplierCompanyName') or {}
        address = self.__data.get('supplierAddress') or {}
        phone = self.__data.get('supplierPhoneNumber') or {}
        tax_id = self.__data.get('supplierBusinessNumber') or {}
        email = self.__data.get('supplierEmail') or {}
        fax = self.__data.get('supplierFax') or {}
        website = self.__data.get("supplierWebsite") or {}
        vat = self.__data.get("supplierVat") or {}

        self.__std_response['merchant_information'] = MerchantInformationInvoice(
                merchant_name=name.get("raw"),
                merchant_address=address.get("raw"),
                merchant_phone=phone.get("raw"),
                merchant_tax_id=tax_id.get("raw"),
                merchant_email=email.get("raw"),
                merchant_fax=fax.get("raw"),
                merchant_website=website.get("raw"),
                vat_number=vat.get("raw"),
                merchant_siren=None,
                merchant_siret=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
        )

        return self.__std_response['merchant_information']

    def std_customer_information(self) -> CustomerInformationInvoice:
        name = self.__data.get("customerCompanyName") or {}
        id = self.__data.get("customerNumber") or {}
        email = self.__data.get("customerEmail") or {}
        tax_id = self.__data.get("customerBusinessNumber") or {}
        billing_address = self.__data.get("customerBillingAddress") or {}
        shipping_address = self.__data.get("customerDeliveryAddress") or {}
        vat = self.__data.get("customerVat") or {}

        self.__std_response['customer_information'] = CustomerInformationInvoice(
                customer_name=name.get("raw"),
                customer_address=billing_address.get("raw"),
                customer_billing_address=billing_address.get("raw"),
                customer_id=id.get("raw"),
                customer_shipping_address=shipping_address.get("raw"),
                customer_email=email.get("raw"),
                vat_number=vat.get("raw"),
                customer_tax_id=tax_id.get("raw"),
                customer_mailing_address=None,
                customer_remittance_address=None,
                customer_service_address=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
            )

        return self.__std_response['customer_information']

    def std_invoice_informations(self) -> None:
        number = self.__data.get("invoiceNumber") or {}
        total = self.__data.get("paymentAmountTotal") or {}
        subtotal = self.__data.get("paymentAmountBase") or {}
        payment_term = self.__data.get("paymentReference") or {}
        amount_due = self.__data.get("paymentAmountDue") or {}
        purchase_order = self.__data.get("invoicePurchaseOrderNumber") or {}

        self.__std_response['invoice_number'] = number.get("raw")
        self.__std_response['invoice_total'] = convert_string_to_number(total.get("raw"), float)
        self.__std_response['invoice_subtotal'] = convert_string_to_number(subtotal.get("parsed"), float)
        self.__std_response['payment_term'] = payment_term.get("raw")
        self.__std_response['amount_due'] = convert_string_to_number(amount_due.get("raw"), float)
        self.__std_response['purchase_order_number'] = purchase_order.get("raw")

    def std_dates_informations(self) -> Tuple[Optional[str], Optional[str]]:
        date = self.__data.get("invoiceDate") or {}
        time = self.__data.get("invoiceTime") or {}

        due_date = self.__data.get("paymentDateDue") or {}
        due_time = self.__data.get("paymentTimeDue") or {}
        
        self.__std_response['due_date'] = combine_date_with_time(due_date.get("raw"), due_time.get("raw"))
        self.__std_response['date'] = combine_date_with_time(date.get("raw"), time.get("raw"))

        return (self.__std_response['due_date'], self.__std_response['date'])

    def std_bank_information(self) -> BankInvoice:
        iban = self.__data.get("bankIban") or {}
        swift = self.__data.get("bankSwift") or {}
        bsb = self.__data.get("bankBsb") or {}
        sort_code = self.__data.get("bankSortCode") or {}
        account_number = self.__data.get("bankAccountNumber") or {}

        self.__std_response['bank_informations'] = BankInvoice(
            iban=iban.get("raw"),
            swift=swift.get("raw"),
            bsb=bsb.get("raw"),
            sort_code=sort_code.get("raw"),
            account_number=account_number.get("raw"),
            vat_number=None,
            rooting_number=None,
        )

        return self.__std_response['bank_informations']

    def std_taxes_informations(self) -> List[TaxesInvoice]:
        taxe = self.__data.get("paymentAmountTax") or {}

        self.__std_response['taxes'] = [
            TaxesInvoice(
                value=convert_string_to_number(taxe.get("parsed"), float),
                rate=None
            )
        ]

        return self.__std_response['taxes']

    def std_items_lines_informations(self) -> Sequence[ItemLinesInvoice]:
        tables = self.__data.get("tables") or []
        self.__std_response['item_lines'] = []
        for table in tables:
            parsed = table.get("parsed") or {}
            rows = parsed.get("rows") or []
            for item in rows:
                item_parsed = item.get("parsed") or {}
                description = item_parsed.get("itemDescription") or {}
                quantity = item_parsed.get("itemQuantity") or {}
                total = item_parsed.get("itemTotal") or {}
                unit_price = item_parsed.get("itemUnitPrice") or {}

                self.__std_response['item_lines'].append(
                    ItemLinesInvoice(
                        unit_price=convert_string_to_number(unit_price.get("raw"), float),
                        quantity=quantity.get("parsed"),
                        amount=convert_string_to_number(total.get("raw"), float),
                        description=description.get("raw"),
                        date_item=None,
                        product_code=None,
                        tax_item=None, 
                    )
                )

        return self.__std_response['item_lines']

    @property
    def standardized_response(self):
        return InvoiceParserDataClass(
            extracted_data=[
            InfosInvoiceParserDataClass(
                **self.__std_response
                )
            ]
        )


class ReceiptStandardizer:
    __document: Document
    __data: Dict[str, Any]
    __meta: DocumentMeta
    __error: DocumentError
    __std_response: Dict[str, Any]

    def __init__(self, document: Document) -> None:
        self.__document = document
        self.__data = self.__document.data
        self.__error = self.__document.error
        self.__meta = self.__document.meta
        self.__std_response = {}

    @property
    def standardized_response(self):
        return ReceiptParserDataClass(
            extracted_data=[
                InfosReceiptParserDataClass(
                    **self.__std_response
                )
            ]
        )

    def std_merchant_informations(self) -> MerchantInformation:
        name = self.__data.get("supplierCompanyName") or {}
        email = self.__data.get("supplierAddress") or {}
        phone = self.__data.get("supplierPhoneNumber") or {}
        url = self.__data.get("supplierWebsite") or {}
        self.__std_response['merchant_information'] = MerchantInformation(
            merchant_name=name.get("raw"),
            merchant_address=email.get("raw"),
            merchant_phone=phone.get("raw"),
            merchant_url=url.get("raw"),
            merchant_siret=None,
            merchant_siren=None,

        )

        return self.__std_response['merchant_information']

    def std_payment_informations(self) -> PaymentInformation:
        card_number = self.__data.get("paymentCardInformation") or {}
        change = self.__data.get("paymentChange") or {}

        self.__std_response['payment_information'] = PaymentInformation(
            card_type=None,
            card_number=card_number.get("raw"),
            change=change.get("raw"),
            cash=None,
            tip=None,
            discount=None
        )

        return self.__std_response['payment_information']

    def std_locale_information(self) -> Locale:
        currency_code = self.__data.get("receiptCurrencyCode") or {}
        currency_parsed = currency_code.get("parsed") or {}

        self.__std_response['locale'] = Locale(
            currency=currency_parsed.get("value")
        )

        return self.__std_response['locale']

    def std__taxes_informations(self) -> Sequence[Taxes]:
        self.__std_response['taxes'] = []

        for tax in self.__data.get("paymentAmountTax", []):
            self.__std_response['taxes'].append(
                Taxes(
                    taxes=tax.get("raw"),
                    rate=None
                )
            )

        return self.__std_response['taxes']

    def std_item_lines(self) -> List[ItemLines]:
        tables = self.__data.get("lineItemTable") or []
        self.__std_response['item_lines'] = []
        for table in tables:
            parsed = table.get("parsed") or {}
            rows = parsed.get("rows") or []
            for item in rows:
                item_parsed = item.get("parsed") or {}
                description = item_parsed.get("itemDescription") or {}
                quantity = item_parsed.get("itemQuantity") or {}
                total = item_parsed.get("itemTotal") or {}
                unit_price = item_parsed.get("itemUnitPrice") or {}

                self.__std_response['item_lines'].append(
                    ItemLines(
                        unit_price=convert_string_to_number(unit_price.get("raw"), float),
                        quantity=quantity.get("parsed"),
                        amount=convert_string_to_number(total.get("raw"), float),
                        description=description.get("raw"),
                    )
                )

        return self.__std_response['item_lines']

    def std_miscellaneous(self) -> None:
        number = self.__data.get("receiptNumber") or {}
        total = self.__data.get("paymentAmountTotal") or {}
        subtotal = self.__data.get("paymentAmountBAse") or {}
        date = self.__data.get("date") or {}
        time = self.__data.get("time") or {}

        self.__std_response['invoice_number'] = number.get("raw")
        self.__std_response['invoice_total'] = convert_string_to_number(total.get("raw"), float)
        self.__std_response['invoice_subtotal'] = convert_string_to_number(subtotal.get("raw"), float)
        self.__std_response['date'] = date.get("raw")
        self.__std_response['time'] = time.get("raw")


class IdentityStandardizer:
    __document: Document
    __data: Dict[str, Any]
    __meta: DocumentMeta
    __error: DocumentError
    __std_response: Dict[str, Any]

    def __init__(self, document: Document) -> None:
        self.__document = document
        self.__data = self.__document.data
        self.__error = self.__document.error
        self.__meta = self.__document.meta
        self.__std_response = {}

    @property
    def standardized_response(self):
        return IdentityParserDataClass(
            extracted_data=[
               InfosIdentityParserDataClass(
                    **self.__std_response
                ) 
            ]
        )

    def std_names_information(self) -> None:
        last_name = self.__data.get("familyName") or {}
        name = self.__data.get("givenName") or {}
        middle_names = self.__data.get("middle_names") or []

        self.__std_response['last_name'] = ItemIdentityParserDataClass(
            value=last_name.get("raw"),
            confidence=last_name.get("confidence")
        )

        self.__std_response['given_names'] = [
            ItemIdentityParserDataClass(
                value=name.get("raw"),
                confidence=name.get("confidence")
            )
        ]

        for middle_name in middle_names:
            self.__std_response['given_names'].append(
                ItemIdentityParserDataClass(
                    value=middle_name.get("raw"), 
                    confidence=middle_name.get("confidence")
                )
            )

    def std_document_information(self) -> None:
        expiry_date = self.__data.get("expiryDate") or {}
        issuance_date = self.__data.get("issueDate") or {}
        document_id = self.__data.get("documentNumber") or {}
        issuing_state = self.__data.get("authority") or {}
        document_type = self.__data.get("type") or {}
        mrz = self.__data.get("machineReadableZone") or {}

        self.__std_response['expire_date'] = ItemIdentityParserDataClass(
            value=format_date(expiry_date.get("raw")),
            confidence=expiry_date.get("confidence")
        )

        self.__std_response["issuance_date"] = ItemIdentityParserDataClass(
            value=format_date(issuance_date.get("raw")),
            confidence=issuance_date.get("confidence")
        )

        self.__std_response['document_id'] = ItemIdentityParserDataClass(
            value=document_id.get("raw"),
            confidence=document_id.get("confidence")
        )

        self.__std_response['issuing_state'] = ItemIdentityParserDataClass(
            value=issuing_state.get("raw"),
            confidence=issuing_state.get("confidence")
        )

        self.__std_response['document_type'] = ItemIdentityParserDataClass(
            value=document_type.get("raw"),
            confidence=document_type.get("confidence")
        )

        self.__std_response['mrz'] = ItemIdentityParserDataClass(
            value=mrz.get("raw"),
            confidence=mrz.get("confidence")
        )

    def std_location_information(self):
        birth_place  = self.__data.get("birthPlace") or {}
        birth_date = self.__data.get("birthDate") or {}
        country = self.__data.get("issuingCode") or {}
        gender = self.__data.get("sex") or {}
        nationality = self.__data.get("nationality") or {}

        self.__std_response['birth_place'] = ItemIdentityParserDataClass(
            value=birth_place.get("raw"),
            confidence=birth_place.get("confidence")
        )

        self.__std_response['birth_date'] = ItemIdentityParserDataClass(
            value=format_date(birth_date.get("raw")),
            confidence=birth_date.get("confidence")
        )

        self.__std_response['country'] = get_info_country(
            key=InfoCountry.ALPHA3,
            value=country.get("raw", ""),
        )

        self.__std_response['gender'] = ItemIdentityParserDataClass(
            value=gender.get("raw"),
            confidence=gender.get("confidence")
        )

        self.__std_response['nationality'] = ItemIdentityParserDataClass(
            value=nationality.get("raw"),
            confidence=nationality.get("confidence")
        )

        self.__std_response['address'] = ItemIdentityParserDataClass()
        self.__std_response['age'] = ItemIdentityParserDataClass()

