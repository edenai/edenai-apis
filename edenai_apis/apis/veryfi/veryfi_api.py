from http import HTTPStatus
from io import BufferedReader
import mimetypes
from typing import Dict
import uuid

import requests
from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    MicrModel,
)
from edenai_apis.features.ocr.bank_check_parsing.bank_check_parsing_dataclass import (
    ItemBankCheckParsingDataClass,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    BankInvoice,
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    BarCode,
    CustomerInformation,
    Locale,
    InfosReceiptParserDataClass,
    ItemLines,
    MerchantInformation,
    PaymentInformation,
    ReceiptParserDataClass,
    Taxes,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import load_key
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException


class VeryfiApi(ProviderInterface, OcrInterface):
    provider_name = "veryfi"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_key(
            provider_name=self.provider_name, api_keys=api_keys
        )
        self.client_id = self.api_settings["client_id"]
        self.client_secret = self.api_settings["client_secret"]
        self.authorization = self.api_settings["Authorization"]
        self.url = "https://api.veryfi.com/api/v8/partner"

        self.headers = {
            "Accept": "application/json",
            "CLIENT-ID": self.client_id,
            "Authorization": self.authorization,
        }

    def _make_post_request(self, file: BufferedReader, endpoint: str = "/documents"):
        payload = {"file_name": f"test-{uuid.uuid4()}"}

        files = {"file": ("file", file, mimetypes.guess_type(file.name)[0])}

        response = requests.request(
            method="POST",
            url=self.url + endpoint,
            headers=self.headers,
            data=payload,
            files=files,
        )

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException(message=response.json(), code=response.status_code)

        if response.status_code != HTTPStatus.CREATED:
            raise ProviderException(message=response.json(), code=response.status_code)

        return response.json()

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[InvoiceParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)

        file_.close()

        ship_name = original_response["ship_to"]["name"]
        ship_address = original_response["ship_to"]["address"]
        if ship_name is not None and ship_address is not None:
            ship_address = ship_name + ship_address

        customer_information = CustomerInformationInvoice(
            customer_name=original_response["bill_to"]["name"],
            customer_address=original_response["bill_to"]["address"],
            customer_tax_id=original_response["bill_to"]["vat_number"],
            customer_id=original_response["account_number"],
            customer_billing_address=original_response["bill_to"]["address"],
            customer_shipping_address=ship_address,
        )

        merchant_information = MerchantInformationInvoice(
            merchant_name=original_response["vendor"]["name"],
            merchant_address=original_response["vendor"]["address"],
            merchant_phone=original_response["vendor"]["phone_number"],
            merchant_email=original_response["vendor"]["email"],
            merchant_tax_id=original_response["vendor"]["vat_number"],
            merchant_id=original_response["vendor"]["reg_number"],
            merchant_website=original_response["vendor"]["web"],
            merchant_fax=original_response["vendor"]["fax_number"],
        )

        bank_informations = BankInvoice(
            account_number=original_response["vendor"]["account_number"],
            iban=original_response["vendor"]["iban"],
            swift=original_response["vendor"]["bank_swift"],
            vat_number=original_response["vendor"]["vat_number"],
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
                taxes=[TaxesInvoice(value=original_response["tax"])],
                invoice_total=original_response["total"],
                invoice_subtotal=original_response["subtotal"],
                invoice_number=original_response["invoice_number"],
                date=original_response["date"],
                purchase_order=original_response["purchase_order_number"],
                item_lines=item_lines,
                locale=LocaleInvoice(currency=original_response["currency_code"]),
                bank_informations=bank_informations,
            )
        ]

        standardized_response = InvoiceParserDataClass(extracted_data=info_invoice)

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[ReceiptParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)
        file_.close()

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

        standardized_response = ReceiptParserDataClass(extracted_data=info_receipt)

        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__bank_check_parsing(self, file: str, file_url: str = "") -> ResponseType:
        with open(file, "rb") as file_:
            original_response = self._make_post_request(file_, endpoint="/checks")

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
                    account_number=original_response.get("micr", {}).get(
                        "account_number"
                    ),
                    serial_number=original_response.get("micr", {}).get(
                        "serial_number"
                    ),
                    check_number=original_response["check_number"],
                    routing_number=original_response.get("micr", {}).get(
                        "routing_number"
                    ),
                ),
            )
        ]
        return ResponseType[BankCheckParsingDataClass](
            original_response=original_response,
            standardized_response=BankCheckParsingDataClass(extracted_data=items),
        )
