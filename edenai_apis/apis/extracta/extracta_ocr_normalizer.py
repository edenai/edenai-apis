from edenai_apis.features.ocr.resume_parser import (
    ResumeEducation,
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLocation,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeSkill,
    ResumeLang,
    ResumeWorkExp,
    ResumeWorkExpEntry,
)

from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    MicrModel,
    ItemBankCheckParsingDataClass,
)

from edenai_apis.features.ocr.financial_parser import (
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialCustomerInformation,
    FinancialMerchantInformation,
    FinancialPaymentInformation,
    FinancialDocumentInformation,
    FinancialLocalInformation,
    FinancialBankInformation,
    FinancialLineItem,
    FinancialDocumentMetadata,
)

from typing import Optional, Type, Union
import re


def extracta_resume_parser(original_response: dict) -> ResumeParserDataClass:
    # personal infos
    name = ResumePersonalName(
        raw_name=original_response.get("name", None),
        first_name=original_response.get("first_name", None),
        last_name=original_response.get("last_name", None),
        middle=None,
        title=None,
        prefix=None,
        sufix=None,
    )

    address = ResumeLocation(
        formatted_location=original_response.get("address", None),
        postal_code=None,
        region=None,
        country=None,
        country_code=None,
        raw_input_location=None,
        street=None,
        street_number=None,
        appartment_number=None,
        city=None,
    )

    phones = []
    for phone in original_response.get("phones", []):
        phones.append(phone)

    mails = []
    for mail in original_response.get("mails", []):
        mails.append(mail)

    urls = []
    for url in original_response.get("urls", []):
        urls.append(url)

    personal_infos = ResumePersonalInfo(
        name=name,
        address=address,
        self_summary=original_response.get("self_summary", None),
        objective=None,
        date_of_birth=None,
        place_of_birth=None,
        phones=phones,
        mails=mails,
        urls=urls,
        fax=[],
        current_profession=None,
        gender=None,
        nationality=None,
        martial_status=None,
        current_salary=None,
    )

    # education
    education_entries = []
    for edu in original_response.get("education", []):
        education_address = ResumeLocation(
            formatted_location=edu.get("address", None),
            postal_code=None,
            region=None,
            country=None,
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=None,
        )
        education_entries.append(
            ResumeEducationEntry(
                title=edu.get("title", None),
                start_date=edu.get("start_date", None),
                end_date=edu.get("end_date", None),
                location=education_address,
                establishment=edu.get("establishment", None),
                description=edu.get("description", None),
                gpa=edu.get("gpa", None),
                accreditation=edu.get("accreditation", None),
            )
        )
    education = ResumeEducation(
        total_years_education=None,
        entries=education_entries,
    )

    # work experience
    work_experience_entries = []
    for work in original_response.get("work_experience", []):
        work_address = ResumeLocation(
            formatted_location=work.get("address", None),
            postal_code=None,
            region=None,
            country=None,
            country_code=None,
            raw_input_location=None,
            street=None,
            street_number=None,
            appartment_number=None,
            city=None,
        )
        work_experience_entries.append(
            ResumeWorkExpEntry(
                title=work.get("title", None),
                start_date=work.get("start_date", None),
                end_date=work.get("end_date", None),
                company=work.get("company_name", None),
                location=work_address,
                description=work.get("description", None),
                industry=work.get("industry", None),
            )
        )
    work_experience = ResumeWorkExp(
        total_years_experience=None,
        entries=work_experience_entries,
    )

    # language
    languages = []
    for lang in original_response.get("languages", []):
        languages.append(
            ResumeLang(name=lang.get("name", None), code=lang.get("code", None))
        )

    # skills
    skills = []
    for skill in original_response.get("skills", []):
        skills.append(
            ResumeSkill(name=skill.get("name", None), type=skill.get("type", None))
        )

    # certifications
    certifications = []
    for cert in original_response.get("certifications", []):
        certifications.append(
            ResumeSkill(name=cert.get("name", None), type=cert.get("type", None))
        )

    # courses
    courses = []
    for course in original_response.get("courses", []):
        courses.append(
            ResumeSkill(name=course.get("name", None), type=course.get("type", None))
        )

    # publications
    publications = []
    for publication in original_response.get("publications", []):
        publications.append(
            ResumeSkill(
                name=publication.get("name", None), type=publication.get("type", None)
            )
        )

    # interests
    interests = []
    for interest in original_response.get("interests", []):
        interests.append(
            ResumeSkill(
                name=interest.get("name", None), type=interest.get("type", None)
            )
        )

    # extracted data
    extracted_data = ResumeExtractedData(
        personal_infos=personal_infos,
        education=education,
        work_experience=work_experience,
        languages=languages,
        skills=skills,
        certifications=certifications,
        courses=courses,
        publications=publications,
        interests=interests,
    )

    return ResumeParserDataClass(extracted_data=extracted_data)


def extracta_bank_check_parsing(original_response: dict) -> BankCheckParsingDataClass:
    # Convert amount to float if not None
    amount = original_response.get("amount")
    if amount is not None:
        try:
            amount = float(amount)
        except ValueError:
            # Handle or log the error if the conversion fails
            amount = None

    # extracted data
    extracted_data = [
        ItemBankCheckParsingDataClass(
            amount=amount,
            amount_text=original_response.get("amount_text", None),
            bank_address=original_response.get("bank_address", None),
            bank_name=original_response.get("bank_name", None),
            date=original_response.get("date", None),
            memo=original_response.get("memo", None),
            payer_address=original_response.get("payer_address", None),
            payer_name=original_response.get("payer_name", None),
            receiver_address=original_response.get("receiver_address", None),
            receiver_name=original_response.get("receiver_name", None),
            currency=original_response.get("currency", None),
            micr=MicrModel(
                raw=original_response.get("micr", {}).get("micr_raw", None),
                account_number=original_response.get("micr", {}).get(
                    "account_number", None
                ),
                routing_number=original_response.get("micr", {}).get(
                    "routing_number", None
                ),
                serial_number=original_response.get("micr", {}).get(
                    "serial_number", None
                ),
                check_number=original_response.get("micr", {}).get(
                    "check_number", None
                ),
            ),
        )
    ]

    return BankCheckParsingDataClass(extracted_data=extracted_data)


def _format_string_for_conversion(string_number: str) -> str:
    commas_occurences = [match.start() for match in re.finditer("\,", string_number)]
    dot_occurences = [match.start() for match in re.finditer("\.", string_number)]

    if len(commas_occurences) > 0 and len(dot_occurences) > 0:
        index_remove_partt = max(
            commas_occurences[len(commas_occurences) - 1],
            dot_occurences[len(dot_occurences) - 1],
        )
        number_part = string_number[:index_remove_partt]
        degit_part = string_number[index_remove_partt + 1 :]
        number_part = re.sub(r"[^\d]", "", number_part)
        return f"{number_part}.{degit_part}"
    if len(commas_occurences) > 0:
        if len(commas_occurences) == 1:
            return string_number.replace(",", ".")
    if len(dot_occurences) > 0:
        if len(dot_occurences) == 1:
            return string_number
    return re.sub(r"[^\d]", "", string_number)


def convert_string_to_number(
    string_number: Optional[str], val_type: Union[Type[int], Type[float]]
) -> Union[int, float, None]:
    """convert a `string` to either `int` or `float`"""
    if not string_number:
        return None
    if isinstance(string_number, (int, float)):
        return string_number
    if isinstance(string_number, str):
        string_number = string_number.strip()
    try:
        number_nature = 1
        # test if negatif element
        if string_number[0] == "-":
            number_nature = -1
        string_formatted = _format_string_for_conversion(
            re.sub(r"[^\d\.\,]", "", string_number)
        )
        return val_type(float(string_formatted)) * number_nature
    except Exception as exc:
        print(exc)
        return None


def remove_empty_strings(data):
    if isinstance(data, dict):
        return {k: remove_empty_strings(v) for k, v in data.items() if v != ""}
    elif isinstance(data, list):
        return [remove_empty_strings(item) for item in data]
    else:
        return data


def extracta_financial_parser(original_response: dict) -> FinancialParserDataClass:
    original_response = remove_empty_strings(original_response)

    # customer information
    customerInformation = original_response.get("customer_information", {})
    customer_information = FinancialCustomerInformation(
        name=customerInformation.get("name", None),
        id_reference=customerInformation.get("id_reference", None),
        mailling_address=customerInformation.get("mailling_address", None),
        billing_address=customerInformation.get("address", None),
        shipping_address=customerInformation.get("shipping_address", None),
        service_address=customerInformation.get("service_address", None),
        remittance_address=customerInformation.get("remittance_address", None),
        email=customerInformation.get("email", None),
        phone=customerInformation.get("phone", None),
        vat_number=customerInformation.get("vat_number", None),
        abn_number=customerInformation.get("abn_number", None),
        gst_number=customerInformation.get("gst_number", None),
        pan_number=customerInformation.get("pan_number", None),
        business_number=customerInformation.get("business_number", None),
        siret_number=customerInformation.get("siret_number", None),
        siren_number=customerInformation.get("siren_number", None),
        customer_number=customerInformation.get("customer_number", None),
        coc_number=customerInformation.get("coc_number", None),
        fiscal_number=customerInformation.get("fiscal_number", None),
        registration_number=customerInformation.get("registration_number", None),
        tax_id=customerInformation.get("tax_id", None),
        website=customerInformation.get("website", None),
        remit_to_name=customerInformation.get("remit_to_name", None),
        city=customerInformation.get("city", None),
        country=customerInformation.get("country", None),
        house_number=customerInformation.get("house_number", None),
        province=customerInformation.get("province", None),
        street_name=customerInformation.get("street_name", None),
        zip_code=customerInformation.get("zip_code", None),
        municipality=customerInformation.get("municipality", None),
    )

    # merchant information
    merchantInformation = original_response.get("merchant_information", {})
    merchant_information = FinancialMerchantInformation(
        name=merchantInformation.get("name", None),
        address=merchantInformation.get("address", None),
        phone=merchantInformation.get("phone", None),
        tax_id=merchantInformation.get("tax_id", None),
        id_reference=merchantInformation.get("id_reference", None),
        vat_number=merchantInformation.get("vat_number", None),
        abn_number=merchantInformation.get("abn_number", None),
        gst_number=merchantInformation.get("gst_number", None),
        business_number=merchantInformation.get("business_number", None),
        siret_number=merchantInformation.get("siret_number", None),
        siren_number=merchantInformation.get("siren_number", None),
        pan_number=merchantInformation.get("pan_number", None),
        coc_number=merchantInformation.get("coc_number", None),
        fiscal_number=merchantInformation.get("fiscal_number", None),
        email=merchantInformation.get("email", None),
        fax=merchantInformation.get("fax", None),
        website=merchantInformation.get("website", None),
        registration=merchantInformation.get("registration", None),
        city=merchantInformation.get("city", None),
        country=merchantInformation.get("country", None),
        house_number=merchantInformation.get("house_number", None),
        province=merchantInformation.get("province", None),
        street_name=merchantInformation.get("street_name", None),
        zip_code=merchantInformation.get("zip_code", None),
        country_code=merchantInformation.get("country_code", None),
    )

    # payment information
    paymentInformation = original_response.get("payment_information", {})
    payment_information = FinancialPaymentInformation(
        amount_due=convert_string_to_number(
            paymentInformation.get("amount_due", None), float
        ),
        amount_tip=convert_string_to_number(
            paymentInformation.get("amount_tip", None), float
        ),
        amount_shipping=convert_string_to_number(
            paymentInformation.get("amount_shipping", None), float
        ),
        amount_change=convert_string_to_number(
            paymentInformation.get("amount_change", None), float
        ),
        amount_paid=convert_string_to_number(
            paymentInformation.get("amount_paid", None), float
        ),
        total=convert_string_to_number(paymentInformation.get("total", None), float),
        subtotal=convert_string_to_number(
            paymentInformation.get("subtotal", None), float
        ),
        total_tax=convert_string_to_number(
            paymentInformation.get("total_tax", None), float
        ),
        tax_rate=convert_string_to_number(
            paymentInformation.get("tax_rate", None), float
        ),
        discount=convert_string_to_number(
            paymentInformation.get("discount", None), float
        ),
        gratuity=convert_string_to_number(
            paymentInformation.get("gratuity", None), float
        ),
        service_charge=convert_string_to_number(
            paymentInformation.get("service_charge", None), float
        ),
        previous_unpaid_balance=convert_string_to_number(
            paymentInformation.get("previous_unpaid_balance", None), float
        ),
        prior_balance=convert_string_to_number(
            paymentInformation.get("prior_balance", None), float
        ),
        payment_terms=paymentInformation.get("payment_terms", None),
        payment_method=paymentInformation.get("payment_method", None),
        payment_card_number=paymentInformation.get("payment_card_number", None),
        payment_auth_code=paymentInformation.get("payment_auth_code", None),
        shipping_handling_charge=convert_string_to_number(
            paymentInformation.get("shipping_handling_charge", None), float
        ),
        transaction_number=paymentInformation.get("transaction_number", None),
        transaction_reference=paymentInformation.get("transaction_reference", None),
    )

    # financial document information
    financialDocumentInformation = original_response.get(
        "financial_document_information", {}
    )
    financial_document_information = FinancialDocumentInformation(
        invoice_receipt_id=financialDocumentInformation.get("invoice_receipt_id", None),
        purchase_order=financialDocumentInformation.get("purchase_order", None),
        invoice_date=financialDocumentInformation.get("invoice_date", None),
        time=financialDocumentInformation.get("time", None),
        invoice_due_date=financialDocumentInformation.get("invoice_due_date", None),
        service_start_date=financialDocumentInformation.get("service_start_date", None),
        service_end_date=financialDocumentInformation.get("service_end_date", None),
        reference=financialDocumentInformation.get("reference", None),
        biller_code=financialDocumentInformation.get("biller_code", None),
        order_date=financialDocumentInformation.get("order_date", None),
        tracking_number=financialDocumentInformation.get("tracking_number", None),
        barcodes=[],
    )

    # local information
    localInfo = original_response.get("local", {})
    local = FinancialLocalInformation(
        currency=localInfo.get("currency", None),
        currency_code=localInfo.get("currency_code", None),
        currency_exchange_rate=localInfo.get("currency_exchange_rate", None),
        country=localInfo.get("country", None),
        language=localInfo.get("language", None),
    )

    # bank information
    bankInfo = original_response.get("bank", {})
    bank = FinancialBankInformation(
        iban=bankInfo.get("iban", None),
        swift=bankInfo.get("swift", None),
        bsb=bankInfo.get("bsb", None),
        sort_code=bankInfo.get("sort_code", None),
        account_number=bankInfo.get("account_number", None),
        routing_number=bankInfo.get("routing_number", None),
        bic=bankInfo.get("bic", None),
    )

    # item lines
    item_lines = []
    for item in original_response.get("item_lines", []):
        item_lines.append(
            FinancialLineItem(
                tax=convert_string_to_number(item.get("tax", None), float),
                amount_line=convert_string_to_number(
                    item.get("amount_line", None), float
                ),
                description=item.get("description", None),
                quantity=convert_string_to_number(item.get("quantity", None), float),
                unit_price=convert_string_to_number(
                    item.get("unit_price", None), float
                ),
                unit_type=item.get("unit_type", None),
                date=item.get("date", None),
                product_code=item.get("product_code", None),
                purchase_order=item.get("purchase_order", None),
                tax_rate=convert_string_to_number(item.get("tax_rate", None), float),
                base_total=convert_string_to_number(
                    item.get("base_total", None), float
                ),
                sub_total=convert_string_to_number(item.get("sub_total", None), float),
                discount_amount=convert_string_to_number(
                    item.get("discount_amount", None), float
                ),
                discount_rate=convert_string_to_number(
                    item.get("discount_rate", None), float
                ),
                discount_code=item.get("discount_code", None),
                order_number=item.get("order_number", None),
                title=item.get("title", None),
            )
        )

    # document metadata
    documentMetadata = original_response.get("document_metadata", {})
    document_metadata = FinancialDocumentMetadata(
        document_index=documentMetadata.get("document_index", None),
        document_page_number=documentMetadata.get("document_page_number", None),
        document_type=financialDocumentInformation.get("document_type", None),
    )

    # extracted data
    extracted_data = [
        FinancialParserObjectDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            payment_information=payment_information,
            financial_document_information=financial_document_information,
            local=local,
            bank=bank,
            item_lines=item_lines,
            document_metadata=document_metadata,
        )
    ]

    return FinancialParserDataClass(extracted_data=extracted_data)
