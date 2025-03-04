from typing import Any, Dict, List, Optional, Sequence, Tuple

from edenai_apis.features.ocr.financial_parser import (
    FinancialBankInformation,
    FinancialBarcode,
    FinancialCustomerInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialPaymentInformation,
)
from edenai_apis.features.ocr.identity_parser import (
    IdentityParserDataClass,
    InfoCountry,
    InfosIdentityParserDataClass,
    ItemIdentityParserDataClass,
    format_date,
    get_info_country,
)
from edenai_apis.features.ocr.invoice_parser import (
    BankInvoice,
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.receipt_parser import (
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
    PaymentInformation,
    ReceiptParserDataClass,
    Taxes,
)
from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLang,
    ResumeLocation,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
)
from edenai_apis.utils.parsing import extract

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
        name = self.__data.get("candidateName") or {}
        return ResumePersonalName(
            raw_name=name.get("raw"),
            first_name=extract(name, ["parsed", "candidateNameFirst", "parsed"]),
            last_name=extract(name, ["parsed", "candidateNameFamily", "parsed"]),
            middle=extract(name, ["parsed", "candidateNameMiddle", "parsed"]),
            title=extract(name, ["parsed", "candidateNameTitle", "parsed"]),
            sufix=extract(name, ["parsed", "candidateNameSuffix", "parsed"]),
            prefix=None,
        )

    def __std_location(
        self, key: str, which_location: Optional[dict] = None
    ) -> ResumeLocation:
        if which_location is None:
            which_location = self.__data
        location = which_location.get(key) or {}
        return ResumeLocation(
            raw_input_location=location.get("rawInput"),
            postal_code=extract(location, ["parsed", "postalCode"]),
            region=extract(location, ["parsed", "state"]),
            country_code=extract(location, ["parsed", "countryCode"]),
            country=extract(location, ["parsed", "country"]),
            appartment_number=extract(location, ["parsed", "apartmentNumber"]),
            city=extract(location, ["parsed", "city"]),
            street=extract(location, ["parsed", "street"]),
            street_number=extract(location, ["parsed", "streetNumber"]),
            formatted_location=extract(location, ["parsed", "formatted"]),
        )

    def std_personnal_information(self) -> ResumePersonalInfo:
        self.__std_response["personal_infos"] = ResumePersonalInfo(
            name=self.__std_names(),
            address=self.__std_location(key="location"),
            phones=[
                phone["raw"]
                for phone in self.__data.get("phoneNumber", []) or []
                if phone.get("raw")
            ],
            mails=[
                email["parsed"]
                for email in self.__data.get("email", []) or []
                if email.get("parsed")
            ],
            urls=[
                website["raw"]
                for website in self.__data.get("website", []) or []
                if website.get("raw")
            ],
            self_summary=extract(self.__data, ["summary", "parsed"]),
            current_profession=self.__data.get("profession"),
            objective=extract(self.__data, ["objective", "parsed"]),
            date_of_birth=extract(self.__data, ["dateOfBirth", "parsed"]),
            place_of_birth=None,
            gender=None,
            nationality=extract(self.__data, ["nationality", "parsed"]),
            martial_status=None,
            current_salary=None,
            availability=extract(self.__data, ["availability", "parsed"]),
        )

        return self.__std_response["personal_infos"]

    def std_education(self) -> ResumeEducation:
        edu_entries: List[ResumeEducationEntry] = []
        for i in self.__data.get("education") or []:
            education = i.get("parsed", {})
            location = self.__std_location(
                which_location=education, key="educationLocation"
            )
            start_date = extract(
                education, ["educationDateRange", "parsed", "start", "date"]
            )
            end_date = extract(
                education, ["educationDateRange", "parsed", "end", "date"]
            )
            grade = extract(education, ["educationGrade", "raw"])
            accreditation = extract(education, ["educationAccreditation", "parsed"])
            establishment = extract(education, ["educationOrganization", "parsed"])
            title = extract(education, ["educationLevel", "parsed", "value"])
            description = extract(education, ["educationMajor", "parsed"])
            edu_entries.append(
                ResumeEducationEntry(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    establishment=establishment,
                    gpa=grade,
                    accreditation=accreditation,
                    title=title,
                    description=description,
                )
            )

        self.__std_response["education"] = ResumeEducation(
            entries=edu_entries, total_years_education=None
        )

        return self.__std_response["education"]

    def std_work_experience(self) -> ResumeWorkExp:
        work_entries: List[ResumeWorkExpEntry] = []
        for i in self.__data.get("workExperience") or []:
            work_experience = i.get("parsed", {})
            work_entries.append(
                ResumeWorkExpEntry(
                    title=extract(work_experience, ["jobTitle", "parsed"]),
                    company=extract(
                        work_experience, ["workExperienceOrganization", "parsed"]
                    ),
                    start_date=extract(
                        work_experience,
                        ["workExperienceDateRange", "parsed", "start", "date"],
                    ),
                    end_date=extract(
                        work_experience,
                        ["workExperienceDateRange", "parsed", "end", "date"],
                    ),
                    description=extract(work_experience, ["jobDescription", "parsed"]),
                    location=self.__std_location(
                        which_location=work_experience, key="workExperienceLocation"
                    ),
                    type=extract(
                        work_experience, ["workExperienceType", "parsed", "value"]
                    ),
                    industry=None,
                )
            )

        total_years = extract(self.__data, ["totalYearsExperience", "parsed"])

        if isinstance(total_years, (int, float)):
            total_years = str(total_years)
        else:
            total_years = None

        self.__std_response["work_experience"] = ResumeWorkExp(
            total_years_experience=total_years,
            entries=work_entries,
        )

        return self.__std_response["work_experience"]

    def std_skills(self) -> List[ResumeSkill]:
        self.__std_response["skills"] = []
        for i in self.__data.get("skill", []) or []:
            name = extract(i, ["parsed", "name"])
            value = extract(i, ["parsed", "type"])
            if name and value:
                self.__std_response["skills"].append(ResumeSkill(name=name, type=value))

        return self.__std_response["skills"]

    def std_miscellaneous(
        self,
    ) -> Tuple[List[ResumeLang], List[ResumeSkill], List[ResumeSkill]]:
        self.__std_response["languages"] = [
            ResumeLang(
                name=extract(i, ["raw"]),
                code=extract(i, ["parsed", "languageName", "parsed", "value"]),
            )
            for i in self.__data.get("language") or []
        ]
        self.__std_response["certifications"] = [
            ResumeSkill(name=extract(i, ["parsed"]), type=None)
            for i in self.__data.get("achievement") or []
        ]
        self.__std_response["publications"] = [
            ResumeSkill(name=i.get("raw"), type=None)
            for i in self.__data.get("publications") or []
        ]

        return (
            self.__std_response["languages"],
            self.__std_response["certifications"],
            self.__std_response["publications"],
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
        name = self.__data.get("supplierCompanyName") or {}
        address = self.__data.get("supplierAddress") or {}
        phone = self.__data.get("supplierPhoneNumber") or {}
        tax_id = self.__data.get("supplierBusinessNumber") or {}
        email = self.__data.get("supplierEmail") or {}
        fax = self.__data.get("supplierFax") or {}
        website = self.__data.get("supplierWebsite") or {}
        vat = self.__data.get("supplierVat") or {}

        self.__std_response["merchant_information"] = MerchantInformationInvoice(
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

        return self.__std_response["merchant_information"]

    def std_customer_information(self) -> CustomerInformationInvoice:
        name = self.__data.get("customerCompanyName") or {}
        id = self.__data.get("customerNumber") or {}
        email = self.__data.get("customerEmail") or {}
        tax_id = self.__data.get("customerBusinessNumber") or {}
        billing_address = self.__data.get("customerBillingAddress") or {}
        shipping_address = self.__data.get("customerDeliveryAddress") or {}
        vat = self.__data.get("customerVat") or {}

        self.__std_response["customer_information"] = CustomerInformationInvoice(
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

        return self.__std_response["customer_information"]

    def std_invoice_informations(self) -> None:
        number = self.__data.get("invoiceNumber") or {}
        total = self.__data.get("paymentAmountTotal") or {}
        subtotal = self.__data.get("paymentAmountBase") or {}
        payment_term = self.__data.get("paymentReference") or {}
        amount_due = self.__data.get("paymentAmountDue") or {}
        purchase_order = self.__data.get("invoicePurchaseOrderNumber") or {}

        self.__std_response["invoice_number"] = number.get("raw")
        self.__std_response["invoice_total"] = convert_string_to_number(
            total.get("raw"), float
        )
        self.__std_response["invoice_subtotal"] = convert_string_to_number(
            subtotal.get("parsed"), float
        )
        self.__std_response["payment_term"] = payment_term.get("raw")
        self.__std_response["amount_due"] = convert_string_to_number(
            amount_due.get("raw"), float
        )
        self.__std_response["purchase_order_number"] = purchase_order.get("raw")

    def std_dates_informations(self) -> Tuple[Optional[str], Optional[str]]:
        date = self.__data.get("invoiceDate") or {}
        time = self.__data.get("invoiceTime") or {}

        due_date = self.__data.get("paymentDateDue") or {}
        due_time = self.__data.get("paymentTimeDue") or {}

        self.__std_response["due_date"] = combine_date_with_time(
            due_date.get("raw"), due_time.get("raw")
        )
        self.__std_response["date"] = combine_date_with_time(
            date.get("raw"), time.get("raw")
        )

        return (self.__std_response["due_date"], self.__std_response["date"])

    def std_bank_information(self) -> BankInvoice:
        iban = self.__data.get("bankIban") or {}
        swift = self.__data.get("bankSwift") or {}
        bsb = self.__data.get("bankBsb") or {}
        sort_code = self.__data.get("bankSortCode") or {}
        account_number = self.__data.get("bankAccountNumber") or {}

        self.__std_response["bank_informations"] = BankInvoice(
            iban=iban.get("raw"),
            swift=swift.get("raw"),
            bsb=bsb.get("raw"),
            sort_code=sort_code.get("raw"),
            account_number=account_number.get("raw"),
            vat_number=None,
            rooting_number=None,
        )

        return self.__std_response["bank_informations"]

    def std_taxes_informations(self) -> List[TaxesInvoice]:
        taxe = self.__data.get("paymentAmountTax") or {}

        self.__std_response["taxes"] = [
            TaxesInvoice(
                value=convert_string_to_number(taxe.get("parsed"), float), rate=None
            )
        ]

        return self.__std_response["taxes"]

    def std_items_lines_informations(self) -> Sequence[ItemLinesInvoice]:
        tables = self.__data.get("tables") or []
        self.__std_response["item_lines"] = []
        for table in tables:
            parsed = table.get("parsed") or {}
            rows = parsed.get("rows") or []
            for item in rows:
                item_parsed = item.get("parsed") or {}
                description = item_parsed.get("itemDescription") or {}
                quantity = item_parsed.get("itemQuantity") or {}
                total = item_parsed.get("itemTotal") or {}
                unit_price = item_parsed.get("itemUnitPrice") or {}

                self.__std_response["item_lines"].append(
                    ItemLinesInvoice(
                        unit_price=convert_string_to_number(
                            unit_price.get("raw"), float
                        ),
                        quantity=quantity.get("parsed"),
                        amount=convert_string_to_number(total.get("raw"), float),
                        description=description.get("raw"),
                        date_item=None,
                        product_code=None,
                        tax_item=None,
                    )
                )

        return self.__std_response["item_lines"]

    @property
    def standardized_response(self):
        return InvoiceParserDataClass(
            extracted_data=[InfosInvoiceParserDataClass(**self.__std_response)]
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
            extracted_data=[InfosReceiptParserDataClass(**self.__std_response)]
        )

    def std_merchant_informations(self) -> MerchantInformation:
        name = self.__data.get("supplierCompanyName") or {}
        email = self.__data.get("supplierAddress") or {}
        phone = self.__data.get("supplierPhoneNumber") or {}
        url = self.__data.get("supplierWebsite") or {}
        self.__std_response["merchant_information"] = MerchantInformation(
            merchant_name=name.get("raw"),
            merchant_address=email.get("raw"),
            merchant_phone=phone.get("raw"),
            merchant_url=url.get("raw"),
            merchant_siret=None,
            merchant_siren=None,
        )

        return self.__std_response["merchant_information"]

    def std_payment_informations(self) -> PaymentInformation:
        card_number = self.__data.get("paymentCardInformation") or {}
        change = self.__data.get("paymentChange") or {}

        self.__std_response["payment_information"] = PaymentInformation(
            card_type=None,
            card_number=card_number.get("raw"),
            change=change.get("raw"),
            cash=None,
            tip=None,
            discount=None,
        )

        return self.__std_response["payment_information"]

    def std_locale_information(self) -> Locale:
        currency_code = self.__data.get("receiptCurrencyCode") or {}
        currency_parsed = currency_code.get("parsed") or {}

        self.__std_response["locale"] = Locale(currency=currency_parsed.get("value"))

        return self.__std_response["locale"]

    def std__taxes_informations(self) -> Sequence[Taxes]:
        self.__std_response["taxes"] = []

        for tax in self.__data.get("paymentAmountTax") or []:
            taxes = convert_string_to_number(tax.get("parsed"), float)
            self.__std_response["taxes"].append(Taxes(taxes=taxes, rate=None))

        return self.__std_response["taxes"]

    def std_item_lines(self) -> List[ItemLines]:
        tables = self.__data.get("lineItemTable") or []
        self.__std_response["item_lines"] = []
        for table in tables:
            parsed = table.get("parsed") or {}
            rows = parsed.get("rows") or []
            for item in rows:
                item_parsed = item.get("parsed") or {}
                description = item_parsed.get("itemDescription") or {}
                quantity = item_parsed.get("itemQuantity") or {}
                total = item_parsed.get("itemTotal") or {}
                unit_price = item_parsed.get("itemUnitPrice") or {}

                self.__std_response["item_lines"].append(
                    ItemLines(
                        unit_price=convert_string_to_number(
                            unit_price.get("raw"), float
                        ),
                        quantity=quantity.get("parsed"),
                        amount=convert_string_to_number(total.get("raw"), float),
                        description=description.get("raw"),
                    )
                )

        return self.__std_response["item_lines"]

    def std_miscellaneous(self) -> None:
        number = self.__data.get("receiptNumber") or {}
        total = self.__data.get("paymentAmountTotal") or {}
        subtotal = self.__data.get("paymentAmountBAse") or {}
        date = self.__data.get("date") or {}
        time = self.__data.get("time") or {}

        self.__std_response["invoice_number"] = number.get("raw")
        self.__std_response["invoice_total"] = convert_string_to_number(
            total.get("raw"), float
        )
        self.__std_response["invoice_subtotal"] = convert_string_to_number(
            subtotal.get("raw"), float
        )
        self.__std_response["date"] = date.get("raw")
        self.__std_response["time"] = time.get("raw")


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
            extracted_data=[InfosIdentityParserDataClass(**self.__std_response)]
        )

    def std_names_information(self) -> None:
        last_name = self.__data.get("familyName") or {}
        name = self.__data.get("givenName") or {}
        middle_names = self.__data.get("middle_names") or []

        self.__std_response["last_name"] = ItemIdentityParserDataClass(
            value=last_name.get("raw"), confidence=last_name.get("confidence")
        )

        self.__std_response["given_names"] = [
            ItemIdentityParserDataClass(
                value=name.get("raw"), confidence=name.get("confidence")
            )
        ]

        for middle_name in middle_names:
            self.__std_response["given_names"].append(
                ItemIdentityParserDataClass(
                    value=middle_name.get("raw"),
                    confidence=middle_name.get("confidence"),
                )
            )

    def std_document_information(self) -> None:
        expiry_date = self.__data.get("expiryDate") or {}
        issuance_date = self.__data.get("issueDate") or {}
        document_id = self.__data.get("documentNumber") or {}
        issuing_state = self.__data.get("authority") or {}
        document_type = self.__data.get("type") or {}
        mrz = self.__data.get("machineReadableZone") or {}

        self.__std_response["expire_date"] = ItemIdentityParserDataClass(
            value=format_date(expiry_date.get("raw")),
            confidence=expiry_date.get("confidence"),
        )

        self.__std_response["issuance_date"] = ItemIdentityParserDataClass(
            value=format_date(issuance_date.get("raw")),
            confidence=issuance_date.get("confidence"),
        )

        self.__std_response["document_id"] = ItemIdentityParserDataClass(
            value=document_id.get("raw"), confidence=document_id.get("confidence")
        )

        self.__std_response["issuing_state"] = ItemIdentityParserDataClass(
            value=issuing_state.get("raw"), confidence=issuing_state.get("confidence")
        )

        self.__std_response["document_type"] = ItemIdentityParserDataClass(
            value=document_type.get("raw"), confidence=document_type.get("confidence")
        )

        self.__std_response["mrz"] = ItemIdentityParserDataClass(
            value=mrz.get("raw"), confidence=mrz.get("confidence")
        )

    def std_location_information(self):
        birth_place = self.__data.get("birthPlace") or {}
        birth_date = self.__data.get("birthDate") or {}
        country = self.__data.get("issuingCode") or {}
        gender = self.__data.get("sex") or {}
        nationality = self.__data.get("nationality") or {}

        self.__std_response["birth_place"] = ItemIdentityParserDataClass(
            value=birth_place.get("raw"), confidence=birth_place.get("confidence")
        )

        self.__std_response["birth_date"] = ItemIdentityParserDataClass(
            value=format_date(birth_date.get("raw")),
            confidence=birth_date.get("confidence"),
        )

        self.__std_response["country"] = get_info_country(
            key=InfoCountry.ALPHA3,
            value=country.get("raw", ""),
        )

        self.__std_response["gender"] = ItemIdentityParserDataClass(
            value=gender.get("raw"), confidence=gender.get("confidence")
        )

        self.__std_response["nationality"] = ItemIdentityParserDataClass(
            value=nationality.get("raw"), confidence=nationality.get("confidence")
        )

        self.__std_response["address"] = ItemIdentityParserDataClass()
        self.__std_response["age"] = ItemIdentityParserDataClass()


class FinancialStandardizer:
    __document: Document
    __data: Dict[str, Any]
    __meta: DocumentMeta
    __error: DocumentError

    __std_response: Dict[str, Any]

    def __init__(self, document: Document, original_response: Dict) -> None:
        self.__document = document
        self.__data = self.__document.data
        self.__error = self.__document.error
        self.__meta = self.__document.meta
        self.__std_response = {}
        self.__formatted_data = self.format_data(original_response)

    def format_data(self, original_response) -> List[Dict]:
        """
        Organize affinda response into a more structured output.
        Each element in the list represents a page of the document (e.g., invoice or receipt) with its fields.

        Args:
        - original_response (dict): The parsed Google document.

        Returns:
        - List[Dict]: A list of dictionaries, each containing organized information about a document page.
        """
        page_dict = {}
        new_response = []

        for page_index, page in enumerate(original_response.get("meta").get("pages")):
            grouped_items = []

            for key_name, key_value in original_response.get("data", {}).items():
                if page_index not in page_dict:
                    page_dict[page_index] = {}

                if key_value:
                    if key_name == "tables" or key_name == "lineItemTable":
                        for item in key_value:
                            if item.get("pageIndex") == page_index:
                                grouped_items.append(item)
                    elif isinstance(key_value, dict):
                        if key_value.get("pageIndex") == page_index:
                            page_dict[page_index][key_name] = key_value

                page_dict[page_index]["items"] = grouped_items

        # Convert the dictionary to a list, maintaining the order of pages
        for page_index, page_elements in sorted(page_dict.items()):
            new_response.append(page_elements)

        return new_response

    def std_response(self) -> List[FinancialParserObjectDataClass]:
        extracted_data = []
        for page_idx, invoice in enumerate(self.__formatted_data):
            address_parsed = extract(
                invoice, ["customerBillingAddress", "parsed"], fallback={}
            )

            customer_information = FinancialCustomerInformation(
                name=extract(invoice, ["customerContactName", "raw"]),
                billing_address=extract(invoice, ["customerBillingAddress", "raw"]),
                shipping_address=extract(invoice, ["customerDeliveryAddress", "raw"]),
                country=address_parsed.get("country"),
                zip_code=address_parsed.get("postalCode"),
                city=address_parsed.get("city"),
                street_name=address_parsed.get("street"),
                house_number=address_parsed.get("apartmentNumber"),
                province=address_parsed.get("state"),
                business_number=extract(invoice, ["customerBusinessNumber", "raw"]),
                email=extract(invoice, ["customerEmail", "raw"]),
                id_reference=extract(invoice, ["customerNumber", "raw"]),
                phone=extract(invoice, ["customerPhoneNumber", "raw"]),
                vat_number=extract(invoice, ["customerVat", "raw"]),
            )
            merchant_information = FinancialMerchantInformation(
                address=extract(invoice, ["supplierAddress", "raw"]),
                country=extract(invoice, ["supplierAddress", "parsed", "country"]),
                street_name=extract(invoice, ["supplierAddress", "parsed", "street"]),
                house_number=extract(
                    invoice, ["supplierAddress", "parsed", "apartmentNumber"]
                ),
                city=extract(invoice, ["supplierAddress", "parsed", "city"]),
                zip_code=extract(invoice, ["supplierAddress", "parsed", "postalCode"]),
                province=extract(invoice, ["supplierAddress", "parsed", "state"]),
                business_number=extract(invoice, ["supplierBusinessNumber", "raw"]),
                name=extract(invoice, ["supplierCompanyName", "raw"]),
                email=extract(invoice, ["supplierEmail", "raw"]),
                phone=extract(invoice, ["supplierPhoneNumber", "raw"]),
                vat_number=extract(invoice, ["supplierVat", "raw"]),
                website=extract(invoice, ["supplierWebsite", "raw"]),
            )
            payment_information = FinancialPaymentInformation(
                amount_paid=convert_string_to_number(
                    extract(invoice, ["paymentAmountPaid", "parsed"]), float
                ),
                total_tax=convert_string_to_number(
                    extract(invoice, ["paymentAmountTax", "parsed"]), float
                ),
                total=convert_string_to_number(
                    extract(invoice, ["paymentAmountTotal", "parsed"]), float
                ),
                amount_due=convert_string_to_number(
                    extract(invoice, ["paymentAmountDue", "parsed"]), float
                ),
                payment_terms=extract(invoice, ["paymentTerms", "raw"]),
                transaction_reference=extract(invoice, ["paymentReference", "raw"]),
                amount_shipping=convert_string_to_number(
                    extract(invoice, ["paymentDelivery", "parsed"]), float
                ),
                subtotal=convert_string_to_number(
                    extract(invoice, ["paymentAmountBase", "parsed"]), float
                ),
                previous_unpaid_balance=extract(invoice, ["openingBalance", "parsed"]),
            )
            financial_document_information = FinancialDocumentInformation(
                invoice_receipt_id=extract(
                    invoice,
                    ["invoiceNumber", "raw"],
                    fallback=extract(invoice, ["receiptNumber", "raw"]),
                ),
                order_date=extract(invoice, ["invoiceNumber", "raw"]),
                purchase_order=extract(invoice, ["invoicePurchaseOrderNumber", "raw"]),
                invoice_due_date=extract(invoice, ["paymentDateDue", "raw"]),
                invoice_date=extract(
                    invoice,
                    ["invoiceDate", "raw"],
                    fallback=extract(invoice, ["date", "raw"]),
                ),
                biller_code=extract(invoice, ["bpayBillerCode", "raw"]),
                time=extract(invoice, ["time", "raw"]),
            )
            bank = FinancialBankInformation(
                account_number=extract(invoice, ["bankAccountNumber", "raw"]),
                bsb=extract(invoice, ["bankBsb", "raw"]),
                iban=extract(invoice, ["bankIban", "raw"]),
                swift=extract(invoice, ["bankSwift", "raw"]),
                sort_code=extract(invoice, ["bankSortCode", "raw"]),
            )
            local = FinancialLocalInformation(
                currency_code=extract(
                    invoice,
                    ["currencyCode", "parsed", "value"],
                    fallback=extract(
                        invoice, ["receiptCurrencyCode", "parsed", "value"]
                    ),
                ),
            )
            document_metadata = FinancialDocumentMetadata(
                document_page_number=page_idx + 1
            )
            item_lines = []
            tables = invoice.get("items") or []
            for table in tables:
                parsed = table.get("parsed") or {}
                rows = parsed.get("rows") or []
                for item in rows:
                    item_parsed = item.get("parsed") or {}
                    description = item_parsed.get("itemDescription") or {}
                    quantity = item_parsed.get("itemQuantity") or {}
                    total = item_parsed.get("itemTotal") or {}
                    unit_price = item_parsed.get("itemUnitPrice") or {}

                    item_lines.append(
                        FinancialLineItem(
                            unit_price=convert_string_to_number(
                                unit_price.get("raw"), float
                            ),
                            quantity=convert_string_to_number(
                                quantity.get("parsed"), int
                            ),
                            amount=convert_string_to_number(total.get("raw"), float),
                            description=description.get("raw"),
                            amount_line=convert_string_to_number(
                                total.get("parsed"), float
                            ),
                        )
                    )

            extracted_data.append(
                FinancialParserObjectDataClass(
                    customer_information=customer_information,
                    merchant_information=merchant_information,
                    payment_information=payment_information,
                    financial_document_information=financial_document_information,
                    bank=bank,
                    local=local,
                    document_metadata=document_metadata,
                    item_lines=item_lines,
                )
            )
        self.__std_response["extracted_data"] = extracted_data
        return self.__std_response["extracted_data"]

    @property
    def standardized_response(self):
        return FinancialParserDataClass(
            extracted_data=[
                invoice for invoice in self.__std_response["extracted_data"]
            ]
        )
