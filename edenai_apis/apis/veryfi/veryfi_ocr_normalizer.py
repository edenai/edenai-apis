from typing import Dict, List

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
    BarCode,
    Taxes,
)
from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    ItemBankCheckParsingDataClass,
    MicrModel,
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


# *****************************Invoice parser***************************************************
def veryfi_invoice_parser(original_response: dict) -> InvoiceParserDataClass:
    ship_name = original_response["ship_to"]["name"]
    ship_address = original_response["ship_to"]["address"]
    if ship_name is not None and ship_address is not None:
        ship_address = ship_name + ship_address

    customer_information = CustomerInformationInvoice(
        customer_name=original_response["bill_to"]["name"],
        customer_address=original_response["bill_to"]["address"],
        customer_email=None,
        customer_id=original_response["account_number"],
        customer_tax_id=original_response["bill_to"]["vat_number"],
        customer_mailing_address=None,
        customer_billing_address=original_response["bill_to"]["address"],
        customer_shipping_address=ship_address,
        customer_service_address=None,
        customer_remittance_address=None,
        abn_number=None,
        gst_number=None,
        pan_number=None,
        vat_number=None,
    )

    merchant_information = MerchantInformationInvoice(
        merchant_name=original_response["vendor"]["name"],
        merchant_address=original_response["vendor"]["address"],
        merchant_phone=original_response["vendor"]["phone_number"],
        merchant_email=original_response["vendor"]["email"],
        merchant_tax_id=original_response["vendor"]["vat_number"],
        merchant_website=original_response["vendor"]["web"],
        merchant_fax=original_response["vendor"]["fax_number"],
        merchant_siren=None,
        merchant_siret=None,
        abn_number=None,
        gst_number=None,
        pan_number=None,
        vat_number=None,
    )

    bank_informations = BankInvoice(
        account_number=original_response["vendor"]["account_number"],
        iban=original_response["vendor"]["iban"],
        swift=original_response["vendor"]["bank_swift"],
        vat_number=original_response["vendor"]["vat_number"],
        bsb=None,
        sort_code=None,
        rooting_number=None,
    )

    item_lines = []
    for item in original_response["line_items"]:
        item_lines.append(
            ItemLinesInvoice(
                description=item["description"],
                quantity=item["quantity"],
                discount=item["discount"],
                unit_price=item["price"],
                tax_item=item["tax"],
                tax_rate=item["tax_rate"],
                amount=item["total"],
                date_item=item["date"],
                product_code=item["sku"],
            )
        )

    info_invoice = [
        InfosInvoiceParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            taxes=[TaxesInvoice(value=original_response["tax"], rate=None)],
            invoice_total=original_response["total"],
            invoice_subtotal=original_response["subtotal"],
            invoice_number=original_response["invoice_number"],
            date=original_response["date"],
            purchase_order=original_response["purchase_order_number"],
            item_lines=item_lines,
            locale=LocaleInvoice(
                currency=original_response["currency_code"], language=None
            ),
            bank_informations=bank_informations,
        )
    ]

    return InvoiceParserDataClass(extracted_data=info_invoice)


# *****************************receipt parser***************************************************
def veryfi_receipt_parser(original_response: dict) -> ReceiptParserDataClass:
    customer_information = CustomerInformation(
        customer_name=original_response["bill_to"]["name"],
    )

    merchant_information = MerchantInformation(
        merchant_name=original_response["vendor"]["name"],
        merchant_address=original_response["vendor"]["address"],
        merchant_phone=original_response["vendor"]["phone_number"],
        merchant_url=original_response["vendor"]["web"],
    )

    payment_information = PaymentInformation(
        card_type=original_response["payment"]["type"],
        card_number=original_response["payment"]["card_number"],
    )

    items_lines = []
    for item in original_response["line_items"]:
        items_lines.append(
            ItemLines(
                description=item["description"],
                quantity=item["quantity"],
                unit_price=item["price"],
                amount=item["total"],
            )
        )

    barcodes = [
        BarCode(type=code["type"], value=code["data"])
        for code in original_response.get("barcodes", [])
        if code["data"] is not None and code["type"] is not None
    ]
    info_receipt = [
        InfosReceiptParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            payment_information=payment_information,
            invoice_number=original_response["invoice_number"],
            invoice_subtotal=original_response["subtotal"],
            invoice_total=original_response["total"],
            date=original_response["date"],
            barcodes=barcodes,
            item_lines=items_lines,
            locale=Locale(currency=original_response["currency_code"]),
            taxes=[Taxes(value=original_response["tax"])],
            category=original_response["category"],
        )
    ]

    return ReceiptParserDataClass(extracted_data=info_receipt)


# *****************************bank check parser***************************************************
def veryfi_bank_check_parser(original_response: dict) -> BankCheckParsingDataClass:
    items = [
        ItemBankCheckParsingDataClass(
            amount=original_response["amount"],
            amount_text=original_response["amount_text"],
            bank_name=original_response["bank_name"],
            bank_address=original_response["bank_address"],
            date=original_response["date"],
            memo=original_response["memo"],
            payer_address=original_response["payer_address"],
            payer_name=original_response["payer_name"],
            receiver_name=original_response["receiver_name"],
            receiver_address=original_response["receiver_address"],
            currency=None,
            micr=MicrModel(
                raw=original_response.get("micr", {}).get("raw"),
                account_number=original_response.get("micr", {}).get("account_number"),
                serial_number=original_response.get("micr", {}).get("serial_number"),
                check_number=original_response["check_number"],
                routing_number=original_response.get("micr", {}).get("routing_number"),
            ),
        )
    ]
    return BankCheckParsingDataClass(extracted_data=items)


# *****************************financial parser***************************************************
def veryfi_financial_parser(original_response: dict) -> FinancialParserDataClass:
    customer_information = FinancialCustomerInformation(
        name=original_response["bill_to"]["name"],
        billing_address=original_response["bill_to"]["address"],
        shipping_address=original_response["ship_to"]["address"],
        vat_number=original_response["bill_to"]["vat_number"],
    )
    merchant_information = FinancialMerchantInformation(
        name=original_response["vendor"]["name"],
        abn_number=original_response["vendor"]["abn_number"],
        address=original_response["vendor"]["address"],
        phone=original_response["vendor"]["phone_number"],
        email=original_response["vendor"]["email"],
    )
    payment_information = FinancialPaymentInformation(
        total=original_response["total"],
        amount_due=original_response["total"],
        discount=original_response["discount"],
        payment_card_number=original_response["payment"]["card_number"],
        subtotal=original_response["subtotal"],
        total_tax=original_response["tax"],
        tax_rate=(original_response.get("tax_lines") or [{}])[0].get("rate"),
        amount_tip=original_response["tip"],
    )
    financial_document_information = FinancialDocumentInformation(
        tracking_number=original_response["tracking_number"],
        invoice_date=original_response["date"],
        invoice_due_date=original_response["due_date"],
        invoice_receipt_id=original_response["invoice_number"],
        order_date=original_response["order_date"],
        purchase_order=original_response["purchase_order_number"],
        reference=original_response["reference_number"],
        invoice_type=original_response["document_type"],
        barcodes=[
            FinancialBarcode(type=code["type"], value=code["data"])
            for code in original_response.get("barcodes", [])
            if code["data"] is not None and code["type"] is not None
        ],
    )
    bank = FinancialBankInformation(
        swift=original_response["vendor"]["bank_swift"],
        account_number=original_response["vendor"]["account_number"],
        iban=original_response["vendor"]["iban"],
    )
    local = FinancialLocalInformation(currency=original_response["currency_code"])
    item_lines = []
    for item in original_response["line_items"]:
        item_lines.append(
            FinancialLineItem(
                description=item["description"],
                quantity=(
                    int(item.get("quantity"))
                    if item.get("quantity") is not None
                    else None
                ),
                discount=item["discount"],
                unit_price=item["price"],
                tax_item=item["tax"],
                tax_rate=item["tax_rate"],
                amount_line=item["total"],
                date_item=item["date"],
                product_code=item["sku"],
            )
        )
    financial_document_object = FinancialParserObjectDataClass(
        customer_information=customer_information,
        merchant_information=merchant_information,
        payment_information=payment_information,
        financial_document_information=financial_document_information,
        bank=bank,
        local=local,
        item_lines=item_lines,
        document_metadata=FinancialDocumentMetadata(
            document_type=original_response.get("document_type")
        ),
    )
    return FinancialParserDataClass(extracted_data=[financial_document_object])
