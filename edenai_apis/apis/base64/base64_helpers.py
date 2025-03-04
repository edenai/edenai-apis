from typing import List, Union, Sequence, Type, Dict
import re
from collections import defaultdict
from itertools import zip_longest
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialCustomerInformation,
    FinancialBankInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialPaymentInformation,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    CustomerInformationInvoice,
    MerchantInformationInvoice,
    InfosInvoiceParserDataClass,
    BankInvoice,
    LocaleInvoice,
    ItemLinesInvoice,
    TaxesInvoice,
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    ItemLines,
    ReceiptParserDataClass,
    MerchantInformation,
    InfosReceiptParserDataClass,
    CustomerInformation,
    Locale,
    PaymentInformation,
    Taxes,
)
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
    retreive_first_number_from_string,
)


# *****************************Parsers utils************************************************
def extract_item_lignes(
    data, item_lines_type: Union[Type[ItemLines], Type[ItemLinesInvoice]]
) -> list:
    items_description = [
        value["value"]
        for key, value in data.items()
        if key.startswith("lineItem") and key.endswith("Description")
    ]
    items_quantity = [
        value["value"]
        for key, value in data.items()
        if key.startswith("lineItem") and key.endswith("Quantity")
    ]
    items_unit_price = [
        value["value"]
        for key, value in data.items()
        if key.startswith("lineItem") and key.endswith("UnitPrice")
    ]
    items_total_cost = [
        value["value"]
        for key, value in data.items()
        if key.startswith("lineItem") and key.endswith("LineTotal")
    ]

    items: Sequence[item_lines_type] = []
    for item in zip_longest(
        items_description,
        items_quantity,
        items_total_cost,
        items_unit_price,
        fillvalue=None,
    ):
        item_quantity = retreive_first_number_from_string(
            item[1]
        )  # avoid cases where the quantity is concatenated with a string
        items.append(
            item_lines_type(
                description=item[0] if item[0] else "",
                quantity=convert_string_to_number(item_quantity, float),
                amount=convert_string_to_number(item[2], float),
                unit_price=convert_string_to_number(item[3], float),
            )
        )
    return items


def organize_document_data_by_page(original_response: Dict) -> List[Dict]:
    """
    Base64 parser response is a list of ducments, where each document represents an invoice.
    For each document, the function identifies the page on which each element is located.
    In our standardization, the response is structured by page, specifying the invoice associated with each page using the 'invoice index' field.
    This function transforms the response to organize all extracted data per page.
    """
    new_response = []
    page_dict = {}
    for idx, document in enumerate(original_response):
        grouped_items = {}
        fields = document.get("fields")
        for key_name, key_value in fields.items():
            if not key_value.get("location"):
                continue
            page_index = key_value["location"]["pageNumber"] - 1
            if page_index not in page_dict:
                page_dict[page_index] = {}

            if key_name.startswith("lineItem"):
                item = key_name.split("lineItem")[1].split()
                match = re.match(r"(\d+)([a-zA-Z]+)", item[0])
                if match:
                    index = int(match.group(1))
                    item_value = match.group(2)
                    if index not in grouped_items:
                        grouped_items[index] = {}
                    grouped_items[index].update({item_value: key_value["value"]})
                continue

            page_dict[page_index][key_name] = key_value
            page_dict[page_index]["invoice_index"] = idx + 1
            page_dict[page_index]["item_lines"] = [v for v in grouped_items.values()]

    # Convert the dictionary to a list, maintaining the order of pages
    for page_index, page_elements in sorted(page_dict.items()):
        new_response.append(page_elements)

    return new_response


# *****************************Invoice parser***************************************************
def format_invoice_document_data(original_response: Dict) -> InvoiceParserDataClass:
    fields = original_response[0].get("fields", [])

    items: Sequence[ItemLinesInvoice] = extract_item_lignes(fields, ItemLinesInvoice)

    default_dict = defaultdict(lambda: None)
    # ----------------------Merchant & customer informations----------------------#
    merchant_name = fields.get("companyName", default_dict).get("value")
    merchant_address = fields.get("from", default_dict).get("value")
    customer_name = fields.get("billTo", default_dict).get("value")
    customer_address = fields.get("address", default_dict).get(
        "value"
    )  # DEPRECATED need to be removed
    customer_mailing_address = fields.get("address", default_dict).get("value")
    customer_billing_address = fields.get("billTo", default_dict).get("value")
    customer_shipping_address = fields.get("shipTo", default_dict).get("value")
    customer_remittance_address = fields.get("soldTo", default_dict).get("value")
    # ---------------------- invoice  informations----------------------#
    invoice_number = fields.get("invoiceNumber", default_dict).get("value")
    invoice_total = fields.get("total", default_dict).get("value")
    invoice_total = convert_string_to_number(invoice_total, float)
    invoice_subtotal = fields.get("subtotal", default_dict).get("value")
    invoice_subtotal = convert_string_to_number(invoice_subtotal, float)
    amount_due = fields.get("balanceDue", default_dict).get("value")
    amount_due = convert_string_to_number(amount_due, float)
    discount = fields.get("discount", default_dict).get("value")
    discount = convert_string_to_number(discount, float)
    taxe = fields.get("tax", default_dict).get("value")
    taxe = convert_string_to_number(taxe, float)
    taxes: Sequence[TaxesInvoice] = [TaxesInvoice(value=taxe, rate=None)]
    # ---------------------- payment informations----------------------#
    payment_term = fields.get("paymentTerms", default_dict).get("value")
    purchase_order = fields.get("purchaseOrder", default_dict).get("value")
    date = fields.get("invoiceDate", default_dict).get("value")
    time = fields.get("invoiceTime", default_dict).get("value")
    date = combine_date_with_time(date, time)
    due_date = fields.get("dueDate", default_dict).get("value")
    due_time = fields.get("dueTime", default_dict).get("value")
    due_date = combine_date_with_time(due_date, due_time)
    # ---------------------- bank and local informations----------------------#
    iban = fields.get("iban", default_dict).get("value")
    account_number = fields.get("accountNumber", default_dict).get("value")
    currency = fields.get("currency", default_dict).get("value")

    invoice_parser = InfosInvoiceParserDataClass(
        merchant_information=MerchantInformationInvoice(
            merchant_name=merchant_name,
            merchant_address=merchant_address,
            merchant_email=None,
            merchant_phone=None,
            merchant_website=None,
            merchant_fax=None,
            merchant_siren=None,
            merchant_siret=None,
            merchant_tax_id=None,
            abn_number=None,
            vat_number=None,
            pan_number=None,
            gst_number=None,
        ),
        customer_information=CustomerInformationInvoice(
            customer_name=customer_name,
            customer_address=customer_address,
            customer_email=None,
            customer_id=None,
            customer_mailing_address=customer_mailing_address,
            customer_remittance_address=customer_remittance_address,
            customer_shipping_address=customer_shipping_address,
            customer_billing_address=customer_billing_address,
            customer_service_address=None,
            customer_tax_id=None,
            pan_number=None,
            gst_number=None,
            vat_number=None,
            abn_number=None,
        ),
        invoice_number=invoice_number,
        invoice_total=invoice_total,
        invoice_subtotal=invoice_subtotal,
        amount_due=amount_due,
        discount=discount,
        taxes=taxes,
        payment_term=payment_term,
        purchase_order=purchase_order,
        date=date,
        due_date=due_date,
        locale=LocaleInvoice(
            currency=currency,
            language=None,
        ),
        bank_informations=BankInvoice(
            iban=iban,
            account_number=account_number,
            bsb=None,
            sort_code=None,
            vat_number=None,
            rooting_number=None,
            swift=None,
        ),
        item_lines=items,
    )

    standardized_response = InvoiceParserDataClass(extracted_data=[invoice_parser])

    return standardized_response


# ***************************** Receipt Parser **************************
def format_receipt_document_data(data) -> ReceiptParserDataClass:
    fields = data[0].get("fields", [])

    items: Sequence[ItemLines] = extract_item_lignes(fields, ItemLines)

    default_dict = defaultdict(lambda: None)
    invoice_number = fields.get("receiptNo", default_dict)["value"]
    invoice_total = fields.get("total", default_dict)["value"]
    invoice_total = convert_string_to_number(invoice_total, float)
    date = fields.get("date", default_dict)["value"]
    time = fields.get("time", default_dict)["value"]
    date = combine_date_with_time(date, time)
    invoice_subtotal = fields.get("subtotal", default_dict)["value"]
    invoice_subtotal = convert_string_to_number(invoice_subtotal, float)
    customer_name = fields.get("shipTo", default_dict)["value"]
    merchant_name = fields.get("companyName", default_dict)["value"]
    merchant_address = fields.get("addressBlock", default_dict)["value"]
    currency = fields.get("currency", default_dict)["value"]
    card_number = fields.get("cardNumber", default_dict)["value"]
    card_type = fields.get("cardType", default_dict)["value"]

    taxe = fields.get("tax", default_dict)["value"]
    taxe = convert_string_to_number(taxe, float)
    taxes: Sequence[Taxes] = [Taxes(taxes=taxe)]
    receipt_infos = {
        "payment_code": fields.get("paymentCode", default_dict)["value"],
        "host": fields.get("host", default_dict)["value"],
        "payment_id": fields.get("paymentId", default_dict)["value"],
        "card_type": card_type,
        "receipt_number": invoice_number,
    }

    receipt_parser = InfosReceiptParserDataClass(
        invoice_number=invoice_number,
        invoice_total=invoice_total,
        invoice_subtotal=invoice_subtotal,
        locale=Locale(currency=currency),
        merchant_information=MerchantInformation(
            merchant_name=merchant_name, merchant_address=merchant_address
        ),
        customer_information=CustomerInformation(customer_name=customer_name),
        payment_information=PaymentInformation(
            card_number=card_number, card_type=card_type
        ),
        date=str(date),
        time=str(time),
        receipt_infos=receipt_infos,
        item_lines=items,
        taxes=taxes,
    )

    standardized_response = ReceiptParserDataClass(extracted_data=[receipt_parser])

    return standardized_response


# ***************************** Financial Documents Parser **************************
def format_financial_document_data(original_response: Dict) -> FinancialParserDataClass:
    """
    Formats raw financial document data into a structured format.

    Args:
        original_response (Dict): Raw data extracted with base64.

    Returns:
        FinancialParserDataClass: Structured data object containing financial information.
    """
    # Organize data by page
    formatted_response = organize_document_data_by_page(original_response)
    document_type = original_response[0]["model"]["name"]
    extracted_data: List[FinancialParserObjectDataClass] = []

    # Iterate over organized pages
    for page_index, page_data in enumerate(formatted_response):
        # Extract information from the page
        customer_information = FinancialCustomerInformation(
            name=page_data.get("person", {}).get("value"),
            billing_address=page_data.get("billTo", {}).get("value"),
            shipping_address=page_data.get("shipTo", {}).get("value"),
            remittance_address=page_data.get("soldTo", {}).get("value"),
            phone=page_data.get("phoneNumber", {}).get("value"),
            zip_code=page_data.get("billToPostalCode", {}).get("value"),
            country=page_data.get("billToCountry", {}).get("value"),
            province=page_data.get("billToState", {}).get("value"),
        )

        merchant_information = FinancialMerchantInformation(
            name=page_data.get("companyName", {}).get("value"),
            address=page_data.get("companyAddress", {}).get("value"),
            email=page_data.get("email", {}).get("value"),
            city=page_data.get("companyAddressCity", {}).get("value"),
            province=page_data.get("companyAddressState", {}).get("value"),
            country=page_data.get("companyAddressState", {}).get("value"),
            zip_code=page_data.get("companyAddressPostalCode", {}).get("value"),
            tax_id=page_data.get("vendorTaxId", {}).get("value"),
            vat_number=page_data.get("vatNo", {}).get("value"),
        )

        payment_information = FinancialPaymentInformation(
            total=convert_string_to_number(
                page_data.get("total", {}).get("value"), float
            ),
            subtotal=convert_string_to_number(
                page_data.get("subtotal", {}).get("value"), float
            ),
            discount=convert_string_to_number(
                page_data.get("discount", {}).get("value"), float
            ),
            total_tax=convert_string_to_number(
                page_data.get("tax", {}).get("value"), float
            ),
            amount_due=convert_string_to_number(
                page_data.get("balanceDue", {}).get("value"), float
            ),
            payment_terms=page_data.get("paymentTerms", {}).get("value"),
            payment_method=page_data.get("cardType", {}).get("value"),
            payment_card_number=page_data.get("cardNumber", {}).get("value"),
        )

        financial_document_information = FinancialDocumentInformation(
            invoice_receipt_id=page_data.get("invoiceNumber", {}).get("value")
            or page_data.get("receiptNo", {}).get("value"),
            invoice_date=page_data.get("invoiceDate", {}).get("value")
            or page_data.get("date", {}).get("value"),
            time=page_data.get("time", {}).get("value"),
            purchase_order=page_data.get("purchaseOrder", {}).get("value"),
            invoice_due_date=page_data.get("dueDate", {}).get("value"),
            reference=page_data.get("paymentReference", {}).get("value"),
        )

        extracted_data.append(
            FinancialParserObjectDataClass(
                customer_information=customer_information,
                merchant_information=merchant_information,
                payment_information=payment_information,
                financial_document_information=financial_document_information,
                bank=FinancialBankInformation(
                    iban=page_data.get("iban", {}).get("value"),
                    account_number=page_data.get("accountNumber", {}).get("value"),
                    swift=page_data.get("swiftCode", {}).get("value"),
                ),
                local=FinancialLocalInformation(
                    currency=page_data.get("currency", {}).get("value")
                ),
                document_metadata=FinancialDocumentMetadata(
                    document_index=page_data.get("invoice_index"),
                    document_page_number=page_index + 1,
                    document_type=document_type,
                ),
                item_lines=[
                    FinancialLineItem(
                        tax=convert_string_to_number(item.get("Tax"), float),
                        amount_line=convert_string_to_number(
                            item.get("LineTotal"), float
                        ),
                        description=item.get("Description"),
                        unit_price=convert_string_to_number(
                            item.get("UnitPrice"), float
                        ),
                        quantity=convert_string_to_number(
                            item.get("ShippedQuantity"), int
                        ),
                    )
                    for item in page_data.get("item_lines", [{}])
                ],
            )
        )

    return FinancialParserDataClass(extracted_data=extracted_data)
