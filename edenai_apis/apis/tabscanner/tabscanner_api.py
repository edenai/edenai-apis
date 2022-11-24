from io import BufferedReader
from typing import Any, Sequence
from time import sleep
import requests

from edenai_apis.features import ProviderApi, Ocr
from edenai_apis.features.ocr import (
    ReceiptParserDataClass,
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TabscannerApi(ProviderApi, Ocr):
    provider_name = "tabscanner"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.url = self.api_settings["url"]

    def _process(self, file: BufferedReader, document_type: str) -> str:
        payload = {"documentType": document_type}
        files = {"file": file}
        headers = {"apikey": self.api_key}
        response = requests.post(
            self.url + "2/process", files=files, data=payload, headers=headers
        ).json()
        if response.get("success") == False:
            raise ProviderException(response.get("message"))
        return response["token"]

    def _get_response(self, token: str, retry=0) -> Any:
        headers = {"apikey": self.api_key}
        response = requests.get(self.url + "result/" + token, headers=headers).json()
        if response["status"] == "pending" and retry <= 5:
            sleep(1)
            return self._get_response(token, retry + 1)
        return response

    def ocr__receipt_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[ReceiptParserDataClass]:
        token = self._process(file, "receipt")
        sleep(1)
        original_response = self._get_response(token)

        if "result" not in original_response:
            raise ProviderException(original_response["message"])

        receipt = original_response.get("result")
        address = receipt.get("address")
        barcodes = receipt.get("barcodes")
        date = receipt.get("date")
        currency = receipt.get("currency")
        establishment = receipt.get("establishment")
        payment = receipt.get("paymentMethod")
        subtotal = receipt.get("subTotal")
        total = receipt.get("total")
        phone_number = receipt.get("phoneNumber")

        list_items: Sequence[ItemLines] = [
            ItemLines(
                description=json_element["descClean"],
                amount=float(json_element["lineTotal"])
                if json_element["lineTotal"] and json_element["lineTotal"] != ""
                else None,
                unit_price=float(json_element["unit"])
                if json_element["unit"] and json_element["unit"] != ""
                else None,
                quantity=json_element["qty"],
            )
            for json_element in receipt["lineItems"]
        ]
        receipt_info = {
            "address": address,
            "barcodes": barcodes,
            "Etablishment": establishment,
            "payment": payment,
            "phone_number": phone_number,
        }

        ocr_receipt = InfosReceiptParserDataClass(
            date=date,
            invoice_subtotal=subtotal,
            invoice_total=total,
            merchant_information=MerchantInformation(merchant_name=establishment),
            locale=Locale(currency=currency),
            receipt_infos=receipt_info,
            item_lines=list_items,
        )

        standarized_response = ReceiptParserDataClass(extracted_data=[ocr_receipt])
        result = ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result
