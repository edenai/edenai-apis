from typing import List, Sequence

from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    InfoCountry,
    InfosIdentityParserDataClass,
    ItemIdentityParserDataClass,
    format_date,
    get_info_country,
    IdentityParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    BankInvoice,
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    ItemLinesInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    CustomerInformation,
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
    PaymentInformation,
    Taxes,
)
from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLocation,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)
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
from edenai_apis.utils.parsing import extract


# *****************************Invoice parser***************************************************
def klippa_invoice_parser(original_response: dict) -> InvoiceParserDataClass:
    """
    Parse Klipa original response into an organized invoice parser dataclass object.

    Args:
    - original_response (dict): Klippa original response.

    Returns:
    - FinancialParserDataClass: Parsed invoice data organized into a data class.
    """
    data_response = original_response["data"]
    customer_information = CustomerInformationInvoice(
        customer_name=data_response["customer_name"],
        customer_address=data_response["customer_address"],
        customer_email=data_response["customer_email"],
        customer_phone=data_response["customer_phone"],
        customer_tax_id=data_response["customer_vat_number"],
        customer_id=data_response["customer_id"],
        customer_billing_address=data_response["customer_address"],
        customer_mailing_address=None,
        customer_remittance_address=None,
        customer_service_address=None,
        customer_shipping_address=None,
        abn_number=None,
        vat_number=None,
        gst_number=None,
        pan_number=None,
    )

    merchant_information = MerchantInformationInvoice(
        merchant_name=data_response["merchant_name"],
        merchant_address=data_response["merchant_address"],
        merchant_email=data_response["merchant_email"],
        merchant_phone=data_response["merchant_phone"],
        merchant_tax_id=data_response["merchant_vat_number"],
        merchant_id=data_response["merchant_id"],
        merchant_siret=data_response["merchant_coc_number"],
        merchant_website=data_response["merchant_website"],
        merchant_fax=None,
        merchant_siren=None,
        abn_number=None,
        gst_number=None,
        pan_number=None,
        vat_number=None,
    )

    bank_information = BankInvoice(
        account_number=data_response["merchant_bank_account_number"],
        iban=data_response["merchant_bank_account_number_bic"],
        bsb=None,
        sort_code=data_response["merchant_bank_domestic_bank_code"],
        vat_number=None,
        rooting_number=None,
        swift=None,
    )

    tax_information = TaxesInvoice(
        value=data_response["personal_income_tax_amount"],
        rate=data_response["personal_income_tax_rate"],
    )

    locale_information = LocaleInvoice(
        currency=data_response["currency"],
        language=data_response["document_language"],
    )

    item_lines: List[ItemLinesInvoice] = []
    for line in data_response.get("lines", []):
        for item in line.get("lineitems", []):
            item_lines.append(
                ItemLinesInvoice(
                    description=item["description"],
                    quantity=item["quantity"],
                    unit_price=item["amount_each"],
                    discount=item["discount_amount"],
                    amount=item["amount"],
                    tax_rate=item["vat_percentage"],
                    tax_amount=item["vat_amount"],
                    product_code=item["sku"],
                )
            )

    return InvoiceParserDataClass(
        extracted_data=[
            InfosInvoiceParserDataClass(
                customer_information=customer_information,
                merchant_information=merchant_information,
                bank_informations=bank_information,
                taxes=[tax_information],
                locale=locale_information,
                item_lines=item_lines,
                invoice_number=data_response["invoice_number"],
                invoice_date=data_response["date"],
                invoice_total=data_response["amount"],
            )
        ]
    )


# *****************************receipt parser***************************************************


def klippa_receipt_parser(original_response: dict) -> ReceiptParserDataClass:
    data_response = original_response["data"]
    customer_information = CustomerInformation(
        customer_name=data_response["customer_name"],
    )

    merchant_information = MerchantInformation(
        merchant_name=data_response["merchant_name"],
        merchant_address=data_response["merchant_address"],
        merchant_phone=data_response["merchant_phone"],
        merchant_tax_id=data_response["merchant_vat_number"],
        merchant_siret=data_response["merchant_coc_number"],
        merchant_url=data_response["merchant_website"],
    )

    locale_information = Locale(
        currency=data_response["currency"],
        language=data_response["document_language"],
        country=data_response["merchant_country_code"],
    )

    taxes_information = Taxes(
        rate=data_response["personal_income_tax_rate"],
        taxes=data_response["personal_income_tax_amount"],
    )

    payment_information = PaymentInformation(
        card_type=data_response["paymentmethod"],
        card_number=data_response["payment_card_number"],
    )

    item_lines: List[ItemLines] = []
    for line in data_response.get("lines", []):
        for lineitem in line.get("linetimes", []):
            item_lines.append(
                ItemLines(
                    description=lineitem["description"],
                    quantity=lineitem["quantity"],
                    unit_price=lineitem["amount_each"],
                    amount=lineitem["amount"],
                )
            )

    info_receipt = [
        InfosReceiptParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            locale=locale_information,
            taxes=[taxes_information],
            payment_information=payment_information,
            invoice_number=data_response["invoice_number"],
            date=data_response["date"],
            invoice_total=data_response["amount"],
        )
    ]

    return ReceiptParserDataClass(extracted_data=info_receipt)


# *****************************financial parser***************************************************
def klippa_financial_parser(original_response: dict) -> FinancialParserDataClass:
    """
    Parses data obtained from the Klippa financial parser into a structured format.

    Args:
        original_response (dict): Raw data obtained from the Klippa financial parser.

    Returns:
        FinancialParserDataClass: Structured data object containing financial information.
    """
    data_response = original_response["data"]

    customer_information = FinancialCustomerInformation(
        name=data_response["customer_name"],
        billing_address=data_response["customer_address"],
        email=data_response["customer_email"],
        phone=data_response["customer_phone"],
        vat_number=data_response["customer_vat_number"],
        id_reference=data_response["customer_reference"],
        street_name=data_response["customer_street_name"],
        house_number=data_response["customer_house_number"],
        zip_code=data_response["customer_zipcode"],
        city=data_response["customer_city"],
        province=data_response["customer_municipality"],
        municipality=data_response["customer_municipality"],
        country=data_response["customer_country"],
        coc_number=data_response["customer_coc_number"],
    )

    merchant_information = FinancialMerchantInformation(
        name=data_response["merchant_name"],
        address=data_response["merchant_address"],
        email=data_response["merchant_email"],
        phone=data_response["merchant_phone"],
        vat_number=data_response["merchant_vat_number"],
        id_reference=data_response["merchant_id"],
        coc_number=data_response["merchant_coc_number"],
        website=data_response["merchant_website"],
        street_name=data_response["merchant_street_name"],
        zip_code=data_response["merchant_zipcode"],
        house_number=data_response["merchant_house_number"],
        country=data_response["merchant_country_code"],
        province=data_response["merchant_province"],
        fiscal_number=data_response["merchant_fiscal_number"],
    )

    payment_information = FinancialPaymentInformation(
        total=data_response["amount"],
        subtotal=data_response["amountexvat"],
        amount_due=data_response["amount"],
        amount_change=data_response["amount_change"],
        amount_shipping=data_response["amount_shipping"],
        amount_tip=data_response["amount_tip"],
        tax_rate=data_response["personal_income_tax_rate"],
        total_tax=data_response["vatamount"],
        payment_method=data_response["paymentmethod"],
        payment_card_number=data_response["payment_card_bank"],
        payment_auth_code=data_response["payment_auth_code"],
    )

    financial_document_information = FinancialDocumentInformation(
        invoice_receipt_id=data_response["invoice_number"],
        invoice_date=data_response["date"],
        order_date=data_response["purchasedate"],
        invoice_due_date=data_response["payment_due_date"],
        service_start_date=data_response["service_date"],
        barcodes=(
            [
                FinancialBarcode(value=barcode.get("value"), type=barcode.get("type"))
                for barcode in data_response["barcodes"]
            ]
            if len(data_response.get("barcodes") or []) > 0
            else []
        ),
    )

    bank = FinancialBankInformation(
        account_number=data_response["merchant_bank_account_number"],
        bic=data_response["merchant_bank_account_number_bic"],
        sort_code=data_response["merchant_bank_domestic_bank_code"],
    )

    local = FinancialLocalInformation(
        currency_code=data_response["currency"],
        language=data_response["document_language"],
    )

    item_lines: List[FinancialLineItem] = []
    for line in data_response.get("lines", []):
        for item in line.get("lineitems", []):
            item_lines.append(
                FinancialLineItem(
                    description=item["description"],
                    quantity=item["quantity"],
                    unit_price=item["amount_each"],
                    discount=item["discount_amount"],
                    amount_line=item["amount"],
                    tax_rate=item["vat_percentage"],
                    tax_amount=item["vat_amount"],
                    product_code=item["sku"],
                )
            )

    document_metadata = FinancialDocumentMetadata(
        document_type=data_response["document_type"]
    )

    return FinancialParserDataClass(
        extracted_data=[
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
        ]
    )


# *****************************identity parser***************************************************
def klippa_id_parser(original_response: dict) -> IdentityParserDataClass:
    items: Sequence[InfosIdentityParserDataClass] = []

    parsed_data = original_response.get("data", {}).get("parsed", {})

    issuing_country = get_info_country(
        key=InfoCountry.ALPHA3,
        value=(parsed_data.get("issuing_country") or {}).get("value", ""),
    )

    given_names_dict = parsed_data.get("given_names", {}) or {}
    given_names_string = given_names_dict.get("value", "") or ""
    given_names = given_names_string.split(" ")
    final_given_names = []
    for given_name in given_names:
        final_given_names.append(
            ItemIdentityParserDataClass(
                value=given_name,
                confidence=(parsed_data.get("given_names", {}) or {}).get("confidence"),
            )
        )
    birth_date = parsed_data.get("date_of_birth", {}) or {}
    birth_date_value = birth_date.get("value")
    birth_date_confidence = birth_date.get("confidence")
    formatted_birth_date = format_date(birth_date_value)

    issuance_date = parsed_data.get("date_of_issue", {}) or {}
    issuance_date_value = issuance_date.get("value")
    issuance_date_confidence = issuance_date.get("confidence")
    formatted_issuance_date = format_date(issuance_date_value)

    expire_date = parsed_data.get("date_of_expiry", {}) or {}
    expire_date_value = expire_date.get("value")
    expire_date_confidence = expire_date.get("confidence")
    formatted_expire_date = format_date(expire_date_value)

    last_name = parsed_data.get("surname", {}) or {}
    birth_place = parsed_data.get("place_of_birth", {}) or {}
    document_id = parsed_data.get("document_number", {}) or {}
    issuing_state = parsed_data.get("issuing_institution", {}) or {}

    addr = (parsed_data.get("address", {}) or {}).get("value") or {}
    street = addr.get("house_number", "") or ""
    street += f" {addr.get('street_name', '') or ''}"
    city = addr.get("post_code", "") or ""
    city += f" {addr.get('city', '') or ''}"
    province = addr.get("province", "") or ""
    country = addr.get("country", "") or ""
    formatted_address = ", ".join(
        filter(
            lambda x: x,
            [street.strip(), city.strip(), province.strip(), country.strip()],
        )
    )

    age = parsed_data.get("age", {}) or {}
    document_type = parsed_data.get("document_type", {}) or {}
    gender = parsed_data.get("gender", {}) or {}
    mrz = parsed_data.get("mrz", {}) or {}
    nationality = parsed_data.get("nationality", {}) or {}

    images = []

    if img_value := (parsed_data.get("face", {}) or {}).get("value", ""):
        images.append(
            ItemIdentityParserDataClass(
                value=img_value,
                confidence=None,
            )
        )
    identity_imgs = parsed_data.get("identity_document", []) or []
    if len(identity_imgs) > 0:
        for identity_img in identity_imgs:
            if img_value := identity_img.get("image", ""):
                images.append(
                    ItemIdentityParserDataClass(
                        value=img_value,
                        confidence=None,
                    )
                )

    items.append(
        InfosIdentityParserDataClass(
            last_name=ItemIdentityParserDataClass(
                value=last_name.get("value"),
                confidence=last_name.get("confidence"),
            ),
            given_names=final_given_names,
            birth_place=ItemIdentityParserDataClass(
                value=birth_place.get("value"),
                confidence=birth_place.get("confidence"),
            ),
            birth_date=ItemIdentityParserDataClass(
                value=formatted_birth_date,
                confidence=birth_date_confidence,
            ),
            issuance_date=ItemIdentityParserDataClass(
                value=formatted_issuance_date,
                confidence=issuance_date_confidence,
            ),
            expire_date=ItemIdentityParserDataClass(
                value=formatted_expire_date,
                confidence=expire_date_confidence,
            ),
            document_id=ItemIdentityParserDataClass(
                value=document_id.get("value"),
                confidence=document_id.get("confidence"),
            ),
            issuing_state=ItemIdentityParserDataClass(
                value=issuing_state.get("value"),
                confidence=issuing_state.get("confidence"),
            ),
            address=ItemIdentityParserDataClass(
                value=formatted_address,
            ),
            age=ItemIdentityParserDataClass(
                value=age.get("value"),
                confidence=age.get("confidence"),
            ),
            country=issuing_country,
            document_type=ItemIdentityParserDataClass(
                value=document_type.get("value"),
                confidence=document_type.get("confidence"),
            ),
            gender=ItemIdentityParserDataClass(
                value=gender.get("value"),
                confidence=gender.get("confidence"),
            ),
            image_id=images,
            image_signature=[],
            mrz=ItemIdentityParserDataClass(
                value=mrz.get("value"),
                confidence=mrz.get("confidence"),
            ),
            nationality=ItemIdentityParserDataClass(
                value=nationality.get("value"),
                confidence=nationality.get("confidence"),
            ),
        )
    )

    return IdentityParserDataClass(extracted_data=items)


# *****************************resume parser***************************************************
def klippa_resume_parser(original_response: dict) -> ResumeParserDataClass:
    response_data = original_response.get("data", {}).get("parsed", {})
    applicant = response_data["applicant"]
    name = ResumePersonalName(
        raw_name=extract(applicant, ["name", "value"]),
        first_name=None,
        last_name=None,
        middle=None,
        title=None,
        prefix=None,
        sufix=None,
    )
    address = ResumeLocation(
        formatted_location=None,
        postal_code=None,
        region=None,
        country=extract(applicant, ["address", "country", "value"]),
        country_code=None,
        raw_input_location=None,
        street=None,
        street_number=None,
        appartment_number=None,
        city=extract(applicant, ["address", "city", "value"]),
    )
    phones = extract(applicant, ["phone_number", "value"])
    mails = extract(applicant, ["email_address", "value"])
    urls = [
        website["value"]
        for website in extract(applicant, ["websites"], [])
        if website.get("value")
    ]
    personal_infos = ResumePersonalInfo(
        name=name,
        address=address,
        self_summary=None,
        objective=None,
        date_of_birth=None,
        place_of_birth=None,
        phones=[phones] if phones else [],
        mails=[mails] if mails else [],
        urls=urls,
        fax=[],
        current_profession=None,
        gender=None,
        nationality=None,
        martial_status=None,
        current_salary=None,
    )
    education_entries = []
    for edu in response_data.get("education", []):
        education_address = ResumeLocation(
            formatted_location=None,
            postal_code=None,
            region=None,
            country=extract(edu, ["address", "country", "value"]),
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=extract(edu, ["address", "city", "value"]),
        )
        education_entries.append(
            ResumeEducationEntry(
                title=extract(edu, ["program", "value"]),
                start_date=extract(edu, ["start", "value"]),
                end_date=extract(edu, ["end", "value"]),
                location=education_address,
                establishment=extract(edu, ["institution", "value"]),
                description=extract(edu, ["program", "value"]),
                gpa=None,
                accreditation=None,
            )
        )
    education = ResumeEducation(
        total_years_education=None,
        entries=education_entries,
    )
    work_experience_entries = []
    for work in response_data.get("work_experience", []):
        work_address = ResumeLocation(
            formatted_location=None,
            postal_code=None,
            region=None,
            country=extract(work, ["address", "country", "value"]),
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=extract(work, ["address", "city", "value"]),
        )
        work_experience_entries.append(
            ResumeWorkExpEntry(
                title=extract(work, ["job_title", "value"]),
                start_date=extract(work, ["start", "value"]),
                end_date=extract(work, ["end", "value"]),
                company=extract(work, ["company_name", "value"]),
                location=work_address,
                description=None,
                industry=None,
            )
        )
    work_experience = ResumeWorkExp(
        total_years_experience=None,
        entries=work_experience_entries,
    )
    interests = []
    for interest in response_data.get("other_interests", []):
        interests.append(ResumeSkill(name=interest.get("value"), type=None))
    extracted_data = ResumeExtractedData(
        personal_infos=personal_infos,
        education=education,
        work_experience=work_experience,
        languages=[],
        skills=[],
        certifications=[],
        courses=[],
        publications=[],
        interests=interests,
    )
    return ResumeParserDataClass(extracted_data=extracted_data)
