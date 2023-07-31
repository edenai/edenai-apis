from io import BufferedReader
from typing import Any, Dict, Sequence
from time import sleep
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
            raise ProviderException(response_json.get("message"), code = response.status_code)
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
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[ReceiptParserDataClass]:
        file_ = open(file, "rb")
        token = self._process(file_, "receipt")
        sleep(1)
        original_response, status_code = self._get_response(token)
        file_.close()

        if "result" not in original_response:
            raise ProviderException(original_response["message"], code = status_code)

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
                amount=float(json_element["lineTotal"])
                if json_element["lineTotal"] and json_element["lineTotal"] != ""
                else None,
                unit_price=convert_string_to_number(json_element["unit"], float)
                if json_element["unit"] and json_element["unit"] != ""
                else None,
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
