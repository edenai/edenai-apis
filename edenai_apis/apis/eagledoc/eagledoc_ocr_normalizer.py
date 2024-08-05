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

from collections import defaultdict
from functools import reduce


# **************************************************************************************************
#                                           Helper Functions
# **************************************************************************************************
def dict_get(dictionary, *keys):
    return reduce(
        lambda d, key: d.get(key, None) if isinstance(d, dict) else None,
        keys,
        dictionary,
    )


# **************************************************************************************************
#                                           Invoice Parser
# **************************************************************************************************
def eagledoc_invoice_parser(original_response: dict) -> InvoiceParserDataClass:
    """
    Parse Eagle Doc original response into an organized invoice parser data class object.

    Args:
    - original_response (dict): Eagle Doc original response.

    Returns:
    - InvoiceParserDataClass: Parsed invoice data organized into a data class.
    """

    address_CustomerStreet = dict_get(
        original_response, "general", "CustomerStreet", "value"
    )
    address_CustomerHouseNumber = dict_get(
        original_response, "general", "CustomerHouseNumber", "value"
    )
    address_CustomerCity = dict_get(
        original_response, "general", "CustomerCity", "value"
    )
    address_CustomerZip = dict_get(original_response, "general", "CustomerZip", "value")
    address_CustomerState = dict_get(
        original_response, "general", "CustomerState", "value"
    )
    address_CustomerCountry = dict_get(
        original_response, "general", "CustomerCountry", "value"
    )

    customer_information = CustomerInformationInvoice(
        customer_name=dict_get(original_response, "general", "CustomerName", "value"),
        customer_address=", ".join(
            filter(
                None,
                (
                    address_CustomerStreet,
                    address_CustomerHouseNumber,
                    address_CustomerCity,
                    address_CustomerZip,
                    address_CustomerState,
                    address_CustomerCountry,
                ),
            )
        ),
        customer_email=None,
        customer_tax_id=None,
        customer_id=None,
        customer_billing_address=", ".join(
            filter(
                None,
                (
                    address_CustomerStreet,
                    address_CustomerHouseNumber,
                    address_CustomerCity,
                    address_CustomerZip,
                    address_CustomerState,
                    address_CustomerCountry,
                ),
            )
        ),
        customer_mailing_address=None,
        customer_remittance_address=None,
        customer_service_address=None,
        customer_shipping_address=None,
        abn_number=None,
        vat_number=None,
        pan_number=None,
        siret_number=None,
        siren_number=None,
        gst_number=None,
    )

    address_ShopStreet = dict_get(original_response, "general", "ShopStreet", "value")
    address_ShopHouseNumber = dict_get(
        original_response, "general", "ShopHouseNumber", "value"
    )
    address_ShopCity = dict_get(original_response, "general", "ShopCity", "value")
    address_ShopZip = dict_get(original_response, "general", "ShopZip", "value")
    address_ShopState = dict_get(original_response, "general", "ShopState", "value")
    address_ShopCountry = dict_get(original_response, "general", "ShopCountry", "value")

    merchant_information = MerchantInformationInvoice(
        merchant_name=dict_get(original_response, "general", "ShopName", "value"),
        merchant_address=", ".join(
            filter(
                None,
                (
                    address_ShopStreet,
                    address_ShopHouseNumber,
                    address_ShopZip,
                    address_ShopCity,
                    address_ShopState,
                    address_ShopCountry,
                ),
            )
        ),
        merchant_phone=dict_get(original_response, "general", "ShopTel", "value"),
        merchant_email=dict_get(original_response, "general", "ShopEmail", "value"),
        merchant_fax=None,
        merchant_website=dict_get(original_response, "general", "ShopWeb", "value"),
        merchant_tax_id=dict_get(
            original_response, "general", "ShopTaxNumber", "value"
        ),
        vat_number=dict_get(original_response, "general", "ShopTaxNumber", "value"),
        merchant_siret=None,
        merchant_siren=None,
        abn_number=None,
        gst_number=None,
        pan_number=None,
    )

    paymentBanks = dict_get(original_response, "paymentBanks")

    if paymentBanks is not None:
        iban = dict_get(paymentBanks[0], "IBAN", "value")
        bic = dict_get(paymentBanks[0], "BIC", "value")
    else:
        iban = None
        bic = None

    bank_information = BankInvoice(
        account_number=iban,
        iban=iban,
        bsb=None,
        sort_code=None,
        vat_number=dict_get(original_response, "general", "ShopTaxNumber", "value"),
        rooting_number=None,
        swift=bic,
    )

    taxInfo = dict_get(original_response, "taxes")

    if taxInfo is not None and bool(taxInfo):

        taxes_information = [
            TaxesInvoice(
                value=dict_get(tax, "TaxAmount", "value"),
                rate=dict_get(tax, "TaxPercentage", "value"),
            )
            for tax in taxInfo
        ]

    else:
        taxes_information = []

    locale_information = LocaleInvoice(
        currency=dict_get(original_response, "general", "Currency", "value"),
        language=dict_get(original_response, "mainLanguage"),
    )

    productItems = dict_get(original_response, "productItems")

    if productItems is not None:
        item_lines = [
            ItemLinesInvoice(
                description=dict_get(item, "ProductName", "value"),
                quantity=dict_get(item, "ProductQuantity", "value"),
                unit_price=dict_get(item, "ProductUnitPrice", "value"),
                amount=dict_get(item, "ProductPrice", "value"),
                tax_rate=dict_get(item, "TaxPercentage", "value"),
                tax_amount=dict_get(item, "TaxAmount", "value"),
                product_code=dict_get(item, "ProductId", "value"),
            )
            for item in productItems
        ]
    else:
        item_lines = []

    return InvoiceParserDataClass(
        extracted_data=[
            InfosInvoiceParserDataClass(
                customer_information=customer_information,
                merchant_information=merchant_information,
                bank_informations=bank_information,
                taxes=taxes_information,
                locale=locale_information,
                item_lines=item_lines,
                invoice_number=dict_get(
                    original_response, "general", "InvoiceNumber", "value"
                ),
                date=dict_get(original_response, "general", "InvoiceDate", "value"),
                invoice_total=dict_get(
                    original_response, "general", "TotalPrice", "value"
                ),
                due_date=dict_get(
                    original_response, "general", "InvoiceDueDate", "value"
                ),
                po_number=dict_get(
                    original_response, "general", "OrderNumber", "value"
                ),
            )
        ]
    )


# **************************************************************************************************
#                                           Receipt Parser
# **************************************************************************************************
def eagledoc_receipt_parser(original_response: dict) -> ReceiptParserDataClass:
    """
    Parse Eagle Doc original response into an organized receipt parser data class object.

    Args:
    - original_response (dict): Eagle Doc original response.

    Returns:
    - ReceiptParserDataClass: Parsed receipt data organized into a data class.
    """

    customer_information = CustomerInformation(
        customer_name=dict_get(original_response, "general", "CustomerName", "value"),
    )

    address_ShopStreet = dict_get(original_response, "general", "ShopStreet", "value")
    address_ShopHouseNumber = dict_get(
        original_response, "general", "ShopHouseNumber", "value"
    )
    address_ShopCity = dict_get(original_response, "general", "ShopCity", "value")
    address_ShopZip = dict_get(original_response, "general", "ShopZip", "value")
    address_ShopState = dict_get(original_response, "general", "ShopState", "value")
    address_ShopCountry = dict_get(original_response, "general", "ShopCountry", "value")

    merchant_information = MerchantInformation(
        merchant_name=dict_get(original_response, "general", "ShopName", "value"),
        merchant_address=", ".join(
            filter(
                None,
                (
                    address_ShopStreet,
                    address_ShopHouseNumber,
                    address_ShopZip,
                    address_ShopCity,
                    address_ShopState,
                    address_ShopCountry,
                ),
            )
        ),
        merchant_phone=dict_get(original_response, "general", "ShopTel", "value"),
        merchant_url=dict_get(original_response, "general", "ShopWeb", "value"),
        merchant_tax_id=dict_get(
            original_response, "general", "ShopTaxNumber", "value"
        ),
    )

    locale_information = Locale(
        currency=dict_get(original_response, "general", "Currency", "value"),
        language=dict_get(original_response, "mainLanguage"),
        country=address_ShopCountry,
    )

    taxInfo = dict_get(original_response, "taxes")

    if taxInfo is not None and bool(taxInfo):

        taxes_information = [
            Taxes(
                taxes=dict_get(tax, "TaxAmount", "value"),
                rate=dict_get(tax, "TaxPercentage", "value"),
            )
            for tax in taxInfo
        ]

    else:
        taxes_information = []

    payments = dict_get(original_response, "payments")

    if payments is not None:
        card_type = dict_get(payments[0], "PaymentMethod", "value")
        card_number = dict_get(payments[0], "PaymentCardNumber", "value")
    else:
        card_type = None
        card_number = None

    payment_information = PaymentInformation(
        card_type=card_type,
        card_number=card_number,
    )

    productItems = dict_get(original_response, "productItems")

    if productItems is not None:
        item_lines = [
            ItemLines(
                description=dict_get(item, "ProductName", "value"),
                quantity=dict_get(item, "ProductQuantity", "value"),
                unit_price=dict_get(item, "ProductUnitPrice", "value"),
                amount=dict_get(item, "ProductPrice", "value"),
            )
            for item in productItems
        ]
    else:
        item_lines = []

    date = dict_get(original_response, "general", "Date", "value")
    time = dict_get(original_response, "general", "Time", "value")
    totalPrice = dict_get(original_response, "general", "TotalPrice", "value")

    info_receipt = [
        InfosReceiptParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            locale=locale_information,
            taxes=taxes_information,
            payment_information=payment_information,
            date=date,
            time=time,
            due_date=None,
            item_lines=item_lines,
            invoice_total=totalPrice,
            invoice_subtotal=None,
            barcodes=[],
        )
    ]

    return ReceiptParserDataClass(extracted_data=info_receipt)


# **************************************************************************************************
#                                           Financial Parser
# **************************************************************************************************
def eagledoc_financial_parser(original_response: dict) -> FinancialParserDataClass:
    """
    Parse Eagle Doc original response into an organized financial parser data class object.
    The input can be a receipt or an invoice

    Args:
    - original_response (dict): Eagle Doc original response.

    Returns:
    - FinancialParserDataClass: Parsed financial data organized into a data class.
    """

    address_CustomerStreet = dict_get(
        original_response, "general", "CustomerStreet", "value"
    )
    address_CustomerHouseNumber = dict_get(
        original_response, "general", "CustomerHouseNumber", "value"
    )
    address_CustomerCity = dict_get(
        original_response, "general", "CustomerCity", "value"
    )
    address_CustomerZip = dict_get(original_response, "general", "CustomerZip", "value")
    address_CustomerState = dict_get(
        original_response, "general", "CustomerState", "value"
    )
    address_CustomerCountry = dict_get(
        original_response, "general", "CustomerCountry", "value"
    )

    customer_information = FinancialCustomerInformation(
        name=dict_get(original_response, "general", "CustomerName", "value"),
        id_reference=None,
        mailling_address=None,
        billing_address=", ".join(
            filter(
                None,
                (
                    address_CustomerStreet,
                    address_CustomerHouseNumber,
                    address_CustomerCity,
                    address_CustomerZip,
                    address_CustomerState,
                    address_CustomerCountry,
                ),
            )
        ),
        shipping_address=None,
        service_address=None,
        remittance_address=None,
        email=None,
        phone=None,
        vat_number=None,
        abn_number=None,
        gst_number=None,
        pan_number=None,
        business_number=None,
        siret_number=None,
        siren_number=None,
        customer_number=None,
        coc_number=None,
        fiscal_number=None,
        registration_number=None,
        tax_id=None,
        website=None,
        remit_to_name=None,
        city=address_CustomerCity,
        country=address_CustomerCountry,
        house_number=address_CustomerHouseNumber,
        province=address_CustomerState,
        street_name=address_CustomerStreet,
        zip_code=address_CustomerZip,
        municipality=None,
    )

    address_ShopStreet = dict_get(original_response, "general", "ShopStreet", "value")
    address_ShopHouseNumber = dict_get(
        original_response, "general", "ShopHouseNumber", "value"
    )
    address_ShopCity = dict_get(original_response, "general", "ShopCity", "value")
    address_ShopZip = dict_get(original_response, "general", "ShopZip", "value")
    address_ShopState = dict_get(original_response, "general", "ShopState", "value")
    address_ShopCountry = dict_get(original_response, "general", "ShopCountry", "value")

    merchant_information = FinancialMerchantInformation(
        name=dict_get(original_response, "general", "ShopName", "value"),
        address=", ".join(
            filter(
                None,
                (
                    address_ShopStreet,
                    address_ShopHouseNumber,
                    address_ShopZip,
                    address_ShopCity,
                    address_ShopState,
                    address_ShopCountry,
                ),
            )
        ),
        phone=dict_get(original_response, "general", "Telephone", "value"),
        tax_id=dict_get(original_response, "general", "TaxNumber", "value"),
        id_reference=None,
        vat_number=dict_get(original_response, "general", "VATNumber", "value"),
        abn_number=None,
        gst_number=None,
        business_number=None,
        siret_number=None,
        siren_number=None,
        pan_number=None,
        coc_number=None,
        fiscal_number=None,
        email=dict_get(original_response, "general", "Email", "value"),
        fax=None,
        website=dict_get(original_response, "general", "Website", "value"),
        registration=dict_get(
            original_response, "general", "CompanyRegistrationNumber", "value"
        ),
        city=address_ShopCity,
        country=None,
        house_number=address_ShopHouseNumber,
        province=address_ShopState,
        street_name=address_ShopStreet,
        zip_code=address_ShopZip,
        country_code=address_ShopCountry,
    )

    financial_local_information = FinancialLocalInformation(
        currency=dict_get(original_response, "general", "Currency", "value"),
        currency_code=dict_get(original_response, "general", "Currency", "value"),
        currency_exchange_rate=None,
        country=dict_get(original_response, "general", "ShopCountry", "value"),
        language=dict_get(original_response, "mainLanguage"),
    )

    # calculate the total tax amount
    taxInfo = dict_get(original_response, "taxes")
    totalTaxAmount = 0.0

    if taxInfo is not None and bool(taxInfo):

        for tax in taxInfo:
            try:
                newTaxValue = float(dict_get(tax, "TaxAmount", "value"))
            except TypeError as excp:
                newTaxValue = None
            if newTaxValue is not None:
                totalTaxAmount = totalTaxAmount + newTaxValue

    else:
        totalTaxAmount = None

    # get the first payment method
    payments = dict_get(original_response, "payments")
    paymentMethod = None
    paymentCardNumber = None

    if payments is not None:
        paymentMethod = dict_get(payments[0], "PaymentMethod", "value")
        paymentCardNumber = dict_get(payments[0], "PaymentCardNumber", "value")

    financial_payment_information = FinancialPaymentInformation(
        amount_due=(
            float(dict_get(original_response, "general", "TotalPrice", "value"))
            if dict_get(original_response, "general", "TotalPrice", "value") is not None
            else None
        ),
        amount_tip=(
            float(dict_get(original_response, "general", "Tip", "value"))
            if dict_get(original_response, "general", "Tip", "value") is not None
            else None
        ),
        amount_shipping=None,
        amount_change=None,
        amount_paid=None,
        total=(
            float(dict_get(original_response, "general", "TotalPrice", "value"))
            if dict_get(original_response, "general", "TotalPrice", "value") is not None
            else None
        ),
        subtotal=None,
        total_tax=totalTaxAmount,
        tax_rate=None,
        discount=None,
        gratuity=None,
        service_charge=None,
        previous_unpaid_balance=None,
        prior_balance=None,
        payment_terms=None,
        payment_method=paymentMethod,
        payment_card_number=paymentCardNumber,
        payment_auth_code=dict_get(
            original_response, "general", "TransactionAuthCode", "value"
        ),
        shipping_handling_charge=None,
        transaction_number=dict_get(
            original_response, "general", "Transaction", "value"
        ),
        transaction_reference=dict_get(
            original_response, "general", "TransactionRef", "value"
        ),
    )

    # get bank information
    paymentBanks = dict_get(original_response, "paymentBanks")
    bank_iban = None
    bank_bic = None
    if paymentBanks is not None:
        bank_iban = dict_get(paymentBanks[0], "IBAN", "value")
        bank_bic = dict_get(paymentBanks[0], "BIC", "value")

    financial_bank_information = FinancialBankInformation(
        iban=bank_iban,
        swift=None,
        bsb=None,
        sort_code=None,
        account_number=None,
        routing_number=None,
        bic=bank_bic,
    )

    productItems = dict_get(original_response, "productItems")
    financial_line_items = None

    if productItems is not None:

        financial_line_items = [
            FinancialLineItem(
                tax=None,
                amount_line=(
                    float(dict_get(item, "ProductPrice", "value"))
                    if dict_get(item, "ProductPrice", "value") is not None
                    else None
                ),
                description=dict_get(item, "ProductName", "value"),
                quantity=(
                    float(dict_get(item, "ProductQuantity", "value"))
                    if dict_get(item, "ProductQuantity", "value") is not None
                    else None
                ),
                unit_price=(
                    float(dict_get(item, "ProductUnitPrice", "value"))
                    if dict_get(item, "ProductUnitPrice", "value") is not None
                    else None
                ),
                unit_type=dict_get(item, "ProductUnit", "value"),
                date=None,
                product_code=dict_get(item, "ProductId", "value"),
                purchase_order=None,
                tax_rate=(
                    float(dict_get(item, "TaxPercentage", "value"))
                    if dict_get(item, "TaxPercentage", "value") is not None
                    else None
                ),
                base_total=None,
                sub_total=None,
                discount_amount=None,
                discount_rate=None,
                discount_code=None,
                order_number=None,
                title=dict_get(item, "ProductName", "value"),
            )
            for item in productItems
        ]

    else:
        financial_line_items = []

    # financial_bar_codes = FinancialBarcode(
    #     value=None,
    #     type=None,
    # )

    financial_document_information = FinancialDocumentInformation(
        invoice_receipt_id=dict_get(
            original_response, "general", "InvoiceNumber", "value"
        ),
        purchase_order=dict_get(original_response, "general", "OrderNumber", "value"),
        invoice_date=dict_get(original_response, "general", "InvoiceDate", "value"),
        time=dict_get(original_response, "general", "Time", "value"),
        invoice_due_date=dict_get(
            original_response, "general", "InvoiceDueDate", "value"
        ),
        service_start_date=dict_get(
            original_response, "general", "DeliveryDate1", "value"
        ),
        service_end_date=dict_get(
            original_response, "general", "DeliveryDate2", "value"
        ),
        reference=None,
        biller_code=None,
        order_date=None,
        tracking_number=None,
        # barcodes=financial_bar_codes,
    )

    financial_document_meta_data = FinancialDocumentMetadata(
        document_index=None,
        document_page_number=None,
        document_type=dict_get(original_response, "docType"),
    )

    extracted_data = []

    extracted_data.append(
        FinancialParserObjectDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            payment_information=financial_payment_information,
            financial_document_information=financial_document_information,
            local=financial_local_information,
            bank=financial_bank_information,
            item_lines=financial_line_items,
            document_metadata=financial_document_meta_data,
        )
    )

    return FinancialParserDataClass(extracted_data=extracted_data)
