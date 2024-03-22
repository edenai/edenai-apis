
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
from collections import defaultdict
from functools import reduce

# **************************************************************************************************
#                                           Helper Functions 
# **************************************************************************************************
def dict_get(dictionary, *keys):
    return reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys, dictionary)

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

    address_CustomerStreet = dict_get(original_response, "general", "CustomerStreet", "value")
    address_CustomerHouseNumber = dict_get(original_response, "general", "CustomerHouseNumber", "value")
    address_CustomerCity = dict_get(original_response, "general", "CustomerCity", "value")
    address_CustomerZip = dict_get(original_response, "general", "CustomerZip", "value")
    address_CustomerState = dict_get(original_response, "general", "CustomerState", "value")
    address_CustomerCountry = dict_get(original_response, "general", "CustomerCountry", "value")

    customer_information = CustomerInformationInvoice(
        customer_name = dict_get(original_response, "general", "CustomerName", "value"),
        customer_address = ', '.join(filter(None, (address_CustomerStreet, address_CustomerHouseNumber, address_CustomerCity, address_CustomerZip, address_CustomerState, address_CustomerCountry))),
        customer_email = None,
        customer_tax_id = None,
        customer_id = None,
        customer_billing_address = ', '.join(filter(None, (address_CustomerStreet, address_CustomerHouseNumber, address_CustomerCity, address_CustomerZip, address_CustomerState, address_CustomerCountry))),
        customer_mailing_address=None,
        customer_remittance_address=None,
        customer_service_address=None,
        customer_shipping_address=None,
        abn_number=None,
        vat_number=None,
        pan_number=None,
        siret_number = None,
        siren_number = None,
        gst_number = None,
    )

    address_ShopStreet = dict_get(original_response, "general", "ShopStreet", "value")
    address_ShopHouseNumber = dict_get(original_response, "general", "ShopHouseNumber", "value")
    address_ShopCity = dict_get(original_response, "general", "ShopCity", "value")
    address_ShopZip = dict_get(original_response, "general", "ShopZip", "value")
    address_ShopState = dict_get(original_response, "general", "ShopState", "value")
    address_ShopCountry = dict_get(original_response, "general", "ShopCountry", "value")

    merchant_information = MerchantInformationInvoice(
        merchant_name = dict_get(original_response, "general", "ShopName", "value"),
        merchant_address = ', '.join(filter(None, (address_ShopStreet, address_ShopHouseNumber, address_ShopZip, address_ShopCity, address_ShopState, address_ShopCountry))),
        merchant_phone = dict_get(original_response, "general", "ShopTel", "value"),
        merchant_email = dict_get(original_response, "general", "ShopEmail", "value"),
        merchant_fax = None,
        merchant_website = dict_get(original_response, "general", "ShopWeb", "value"),
        merchant_tax_id = dict_get(original_response, "general", "ShopTaxNumber", "value"),
        vat_number = dict_get(original_response, "general", "ShopTaxNumber", "value"),
        merchant_siret = None,
        merchant_siren = None,
        abn_number = None,
        gst_number = None,
        pan_number = None,
    )

    paymentBanks = dict_get(original_response, "paymentBanks")

    if paymentBanks is not None:
            iban = dict_get(paymentBanks[0], "IBAN", "value")
            bic = dict_get(paymentBanks[0], "BIC", "value")
    else:
            iban = None
            bic = None

    bank_information = BankInvoice(
        account_number = iban,
        iban = iban,
        bsb = None,
        sort_code = None,
        vat_number = dict_get(original_response, "general", "ShopTaxNumber", "value"),
        rooting_number=None,
        swift = bic,
    )

    taxInfo = dict_get(original_response, "taxes")

    if taxInfo is not None and bool(taxInfo):

        taxes_information = [TaxesInvoice(
            value = dict_get(tax, "TaxAmount", "value"),
            rate = dict_get(tax, "TaxPercentage", "value"),
        )
        for tax in taxInfo]
    
    else:
        taxes_information = []

    locale_information = LocaleInvoice(
        currency = dict_get(original_response, "general", "Currency", "value"),
        language = dict_get(original_response, "mainLanguage"),
    )

    productItems = dict_get(original_response, "productItems")

    if productItems is not None:
        item_lines = [
            ItemLinesInvoice(
                description = dict_get(item, "ProductName", "value"),
                quantity = dict_get(item, "ProductQuantity", "value"),
                unit_price = dict_get(item, "ProductUnitPrice", "value"),
                amount = dict_get(item, "ProductPrice", "value"),
                tax_rate = dict_get(item, "TaxPercentage", "value"),
                tax_amount = dict_get(item, "TaxAmount", "value"),
                product_code = dict_get(item, "ProductId", "value"),
            )
            for item in productItems]
    else:
            item_lines = []

    return InvoiceParserDataClass(
        extracted_data=[
            InfosInvoiceParserDataClass(
                customer_information = customer_information,
                merchant_information = merchant_information,
                bank_informations = bank_information,
                taxes = taxes_information,
                locale = locale_information,
                item_lines = item_lines,
                invoice_number = dict_get(original_response, "general", "InvoiceNumber", "value"),
                date = dict_get(original_response, "general", "InvoiceDate", "value"),
                invoice_total = dict_get(original_response, "general", "TotalPrice", "value"),
                due_date = dict_get(original_response, "general", "InvoiceDueDate", "value"),
                po_number = dict_get(original_response, "general", "OrderNumber", "value"),
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
            customer_name = dict_get(original_response, "general", "CustomerName", "value"),
        )

        address_ShopStreet = dict_get(original_response, "general", "ShopStreet", "value")
        address_ShopHouseNumber = dict_get(original_response, "general", "ShopHouseNumber", "value")
        address_ShopCity = dict_get(original_response, "general", "ShopCity", "value")
        address_ShopZip = dict_get(original_response, "general", "ShopZip", "value")
        address_ShopState = dict_get(original_response, "general", "ShopState", "value")
        address_ShopCountry = dict_get(original_response, "general", "ShopCountry", "value")

        merchant_information = MerchantInformation(
            merchant_name = dict_get(original_response, "general", "ShopName", "value"),
            merchant_address = ', '.join(filter(None, (address_ShopStreet, address_ShopHouseNumber, address_ShopZip, address_ShopCity, address_ShopState, address_ShopCountry))),
            merchant_phone = dict_get(original_response, "general", "ShopTel", "value"),
            merchant_url = dict_get(original_response, "general", "ShopWeb", "value"),
            merchant_tax_id = dict_get(original_response, "general", "ShopTaxNumber", "value"),
        )

        locale_information = Locale(
            currency = dict_get(original_response, "general", "Currency", "value"),
            language = dict_get(original_response, "mainLanguage"),
            country=address_ShopCountry,
        )

        taxInfo = dict_get(original_response, "taxes")

        if taxInfo is not None and bool(taxInfo):

            taxes_information = [Taxes(
                taxes = dict_get(tax, "TaxAmount", "value"),
                rate = dict_get(tax, "TaxPercentage", "value"),
            )
            for tax in taxInfo]
        
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
            card_type = card_type,
            card_number = card_number,
        )

        productItems = dict_get(original_response, "productItems")

        if productItems is not None:
            item_lines = [
                ItemLines(
                    description = dict_get(item, "ProductName", "value"),
                    quantity = dict_get(item, "ProductQuantity", "value"),
                    unit_price = dict_get(item, "ProductUnitPrice", "value"),
                    amount = dict_get(item, "ProductPrice", "value"),
                )
                for item in productItems]
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
