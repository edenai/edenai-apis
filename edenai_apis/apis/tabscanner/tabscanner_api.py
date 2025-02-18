from io import BufferedReader
from time import sleep
from typing import Any, Dict, Sequence

import requests

from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr import (
    ReceiptParserDataClass,
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
    PaymentInformation,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
    FinancialCustomerInformation,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialPaymentInformation,
    FinancialBarcode,
    FinancialDocumentInformation,
    FinancialParserObjectDataClass,
    FinancialLineItem,
    FinancialDocumentMetadata,
    FinancialBankInformation,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import BarCode
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import convert_string_to_number
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TabscannerApi(ProviderInterface, OcrInterface):
    provider_name = "tabscanner"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.tabscanner.com/api/"

    def _process(self, file: BufferedReader, document_type: str) -> str:
        payload = {"documentType": document_type}
        files = {"file": file}
        headers = {"apikey": self.api_key}
        response = requests.post(
            self.url + "2/process", files=files, data=payload, headers=headers
        )
        response_json = response.json()
        if response_json.get("success") == False:
            raise ProviderException(
                response_json.get("message"), code=response.status_code
            )
        return response_json["token"]

    def _get_response(self, token: str, retry=0) -> Any:
        headers = {"apikey": self.api_key}
        response = requests.get(self.url + "result/" + token, headers=headers)
        response_json = response.json()
        if response_json["status"] == "pending" and retry <= 5:
            sleep(1)
            return self._get_response(token, retry + 1)
        return response_json, response.status_code

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        with open(file, "rb") as file_:
            token = self._process(file_, "receipt")
            sleep(1)
            original_response, status_code = self._get_response(token)

        if "result" not in original_response:
            raise ProviderException(original_response["message"], code=status_code)

        receipt = original_response.get("result")

        # Merchant information
        merchant_information = MerchantInformation(
            merchant_name=receipt.get("establishment"),
            merchant_address=receipt.get("address"),
            merchant_phone=receipt.get("phoneNumber"),
            merchant_url=receipt.get("url"),
        )
        # Date & time
        date = receipt.get("date")

        # Barcodes
        barcodes = [
            BarCode(type=code_type, value=code_value)
            for code_value, code_type in receipt.get("barcodes", [])
        ]

        # Local
        locale = Locale(currecy=receipt.get("currency"))
        # Payment information
        payment_information = PaymentInformation(
            card_type=str(receipt.get("paymentMethod")),
            cash=str(receipt.get("cash")),
            tip=str(receipt.get("tip")),
            change=str(receipt.get("change")),
            discount=str(receipt.get("discount")),
        )
        # Total & subtotal
        subtotal = receipt.get("subTotal")
        total = receipt.get("total")

        list_items: Sequence[ItemLines] = [
            ItemLines(
                description=json_element["descClean"],
                amount=(
                    float(json_element["lineTotal"])
                    if json_element["lineTotal"] and json_element["lineTotal"] != ""
                    else None
                ),
                unit_price=(
                    convert_string_to_number(json_element["unit"], float)
                    if json_element["unit"] and json_element["unit"] != ""
                    else None
                ),
                quantity=json_element["qty"],
            )
            for json_element in receipt["lineItems"]
        ]
        receipt_info = {
            "address": "",
            "barcodes": barcodes,
            "Etablishment": "",
            "payment": "",
            "phone_number": "",
        }

        ocr_receipt = InfosReceiptParserDataClass(
            invoice_subtotal=subtotal,
            invoice_total=total,
            date=date,
            merchant_information=merchant_information,
            payment_information=payment_information,
            barcodes=barcodes,
            locale=locale,
            receipt_infos=receipt_info,
            item_lines=list_items,
        )

        standardized_response = ReceiptParserDataClass(extracted_data=[ocr_receipt])
        result = ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        with open(file, "rb") as file_:
            token = self._process(file_, document_type)
            sleep(1)
            original_response, status_code = self._get_response(token)

        if "result" not in original_response:
            raise ProviderException(original_response["message"], code=status_code)

        financial_document = original_response.get("result")

        # Merchant information
        merchant_information = FinancialMerchantInformation(
            name=financial_document.get("establishment"),
            address=financial_document.get("address"),
            phone=financial_document.get("phoneNumber"),
            id_reference=financial_document.get("customFields", {}).get("StoreID"),
            website=financial_document.get("url"),
            city=financial_document.get("addressNorm", {}).get("city"),
            house_number=financial_document.get("addressNorm", {}).get("number"),
            province=financial_document.get("addressNorm", {}).get("state"),
            country=financial_document.get("addressNorm", {}).get("country"),
            street_name=financial_document.get("addressNorm", {}).get("street"),
            zip_code=financial_document.get("addressNorm", {}).get("postcode"),
        )
        payment_information = FinancialPaymentInformation(
            amount_due=financial_document.get("total"),
            amount_tip=financial_document.get("tip"),
            total=financial_document.get("total"),
            total_tax=financial_document.get("tax"),
            amount_change=financial_document.get("change"),
            discount=financial_document.get("discount"),
            payment_method=financial_document.get("paymentMethod"),
            subtotal=financial_document.get("subTotal"),
            barcodes=[
                FinancialBarcode(type=code_type, value=code_value)
                for code_value, code_type in financial_document.get("barcodes", [])
            ],
        )
        financial_document_information = FinancialDocumentInformation(
            date=financial_document.get("date")
        )

        local = FinancialLocalInformation(currecy=financial_document.get("currency"))

        list_items = [
            FinancialLineItem(
                description=json_element.get("descClean"),
                amount_line=convert_string_to_number(
                    json_element.get("lineTotal"), float
                ),
                unit_price=convert_string_to_number(json_element.get("unit"), float),
                quantity=json_element["qty"],
                product_code=json_element.get("productCode"),
                discount_amount=json_element.get("discount"),
            )
            for json_element in financial_document["lineItems"]
        ]

        standardized_response = FinancialParserDataClass(
            extracted_data=[
                FinancialParserObjectDataClass(
                    customer_information=FinancialCustomerInformation(),
                    merchant_information=merchant_information,
                    payment_information=payment_information,
                    financial_document_information=financial_document_information,
                    local=local,
                    bank=FinancialBankInformation(),
                    item_lines=list_items,
                    document_metadata=FinancialDocumentMetadata(
                        document_type=financial_document.get("documentType")
                    ),
                )
            ]
        )
        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
