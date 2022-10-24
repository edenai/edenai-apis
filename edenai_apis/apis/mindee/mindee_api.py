from io import BufferedReader
from collections import defaultdict
from typing import List, Sequence, TypeVar
import requests

from edenai_apis.features import ProviderApi, Ocr
from edenai_apis.features.ocr import (
    ReceiptParserDataClass,
    InfosInvoiceParserDataClass,
    InfosReceiptParserDataClass,
    InvoiceParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    CustomerInformationInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    Locale,
    MerchantInformation,
    Taxes,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import combine_date_with_time, convert_string_to_number
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.data_class_manager import DataClassManager
from edenai_apis.utils.types import ResponseType

ParamsApi = TypeVar("ParamsApi")


class MindeeApi(ProviderApi, Ocr):
    provider_name = "mindee"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["ocr_invoice"]["subscription_key"]
        self.url = self.api_settings["ocr_invoice"]["url"]
        self.url_receipt = self.api_settings["ocr_receipt"]["url"]
        self.api_key_receipt = self.api_settings["ocr_receipt"]["subscription_key"]

    def _get_api_attributes(self, file: BufferedReader, language: str) -> ParamsApi:
        params: ParamsApi = {
            "headers": {"Authorization": self.api_key_receipt},
            "files": {"document": file},
            "params": {
                "local": {
                    "langage": language.split("-")[0],
                    "country": language.split("-")[1],
                }
            },
        }
        return params

    def ocr__receipt_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[ReceiptParserDataClass]:
        args = self._get_api_attributes(file, language)
        original_response = requests.post(
            self.url_receipt,
            headers=args["headers"],
            files=args["files"],
            params=args["params"],
        ).json()

        if "document" not in original_response:
            raise ProviderException(
                original_response["api_request"]["error"]["message"]
            )

        receipt_data = original_response["document"]["inference"]["prediction"]

        date = receipt_data["date"]["value"]
        time = receipt_data["time"]["value"]
        date = combine_date_with_time(date, time)
        currency = receipt_data["locale"]["currency"]
        supplier = receipt_data["supplier"]["value"]
        total_value = receipt_data["total_incl"]["value"]
        total = total_value and float(total_value)
        map_keyword_json_to_class = [("value", "taxes"), ("rate", "rate")]
        taxes: List[Taxes] = DataClassManager.from_jsonarray_to_list(
            Taxes, receipt_data["taxes"], map_keyword_json_to_class
        )
        receipt_infos = {
            "language": receipt_data["locale"]["language"],
            "category": receipt_data["category"]["value"],
        }

        ocr_receipt = InfosReceiptParserDataClass(
            invoice_total=total,
            locale=Locale(currency=currency),
            merchant_information=MerchantInformation(merchant_name=supplier),
            date=date,
            taxes=taxes,
            receipt_infos=receipt_infos,
        )
        standarized_response = ReceiptParserDataClass(extracted_data=[ocr_receipt])

        result = ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def ocr__invoice_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[InvoiceParserDataClass]:

        headers = {
            "Authorization": self.api_key,
        }
        files = {"document": file}
        params = {"locale": {"language": language}}
        original_response = requests.post(
            self.url, headers=headers, files=files, params=params
        ).json()

        if "document" not in original_response:
            raise ProviderException(
                original_response["api_request"]["error"]["message"]
            )

        invoice_data = original_response["document"]["inference"]["prediction"]
        default_dict = defaultdict(lambda: None)
        date = invoice_data.get("date", default_dict).get("value", None)
        time = invoice_data.get("time", default_dict).get("value", None)
        date = combine_date_with_time(date, time)
        customer_name = invoice_data.get("customer", default_dict).get(
            "value", None
        )
        customer_address = invoice_data.get("customer_address", default_dict).get(
            "value", None
        )
        merchant_name = invoice_data.get("supplier", default_dict).get(
            "value", None
        )
        merchant_address = invoice_data.get("supplier_address", default_dict).get(
            "value", None
        )
        invoice_total = convert_string_to_number(
            invoice_data.get("total_incl", default_dict).get("value", None), float
        )
        invoice_subtotal = convert_string_to_number(
            invoice_data.get("total_excl", default_dict).get("value", None), float
        )
        due_date = invoice_data.get("due_date", default_dict).get("value", None)
        due_time = invoice_data.get("due_time", default_dict).get("value", None)
        due_date = combine_date_with_time(due_date, due_time)
        invoice_number = invoice_data.get("invoice_number", default_dict).get(
            "value", None
        )
        taxes: Sequence[TaxesInvoice] = [
            TaxesInvoice(value=item.get("value", None), rate=item["rate"])
            for item in invoice_data.get("taxes", [])
        ]
        currency = invoice_data.get("locale", default_dict)["currency"]
        language = invoice_data.get("locale", default_dict)["language"]

        invoice_parser = InfosInvoiceParserDataClass(
            invoice_number=invoice_number,
            invoice_total=invoice_total,
            invoice_subtotal=invoice_subtotal,
            customer_information=CustomerInformationInvoice(
                customer_name=customer_name, customer_address=customer_address
            ),
            merchant_information=MerchantInformationInvoice(
                merchant_name=merchant_name, merchant_address=merchant_address
            ),
            date=date,
            due_date=due_date,
            taxes=taxes,
            locale=LocaleInvoice(currency=currency, language=language),
        )

        standarized_response = InvoiceParserDataClass(
            extracted_data=[invoice_parser]
        ).dict()

        result = ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result
