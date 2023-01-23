from io import BufferedReader
from collections import defaultdict
from typing import List, Optional, Sequence, TypeVar
from pydantic import StrictStr
import requests

from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr import (
    ReceiptParserDataClass,
    InfosInvoiceParserDataClass,
    InfosReceiptParserDataClass,
    InvoiceParserDataClass,
    IdentityParserDataClass,
    InfosIdentityParserDataClass,
    InfoCountry,
    format_date,
    get_info_country,
    ItemIdentityParserDataClass,
)
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    CustomerInformationInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    BankInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    Locale,
    MerchantInformation,
    Taxes,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
    from_jsonarray_to_list
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

ParamsApi = TypeVar("ParamsApi")


class MindeeApi(ProviderInterface, OcrInterface):
    provider_name = "mindee"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["ocr_invoice"]["subscription_key"]
        self.url = self.api_settings["ocr_invoice"]["url"]
        self.url_receipt = self.api_settings["ocr_receipt"]["url"]
        self.api_key_receipt = self.api_settings["ocr_receipt"]["subscription_key"]
        self.url_identity = self.api_settings['ocr_id']['url']

    def _get_api_attributes(self, file: BufferedReader, language: Optional[str] = None) -> ParamsApi:
        params: ParamsApi = {
            "headers": {"Authorization": self.api_key_receipt},
            "files": {"document": file},
            "params": {
                "local": {
                    "langage": language.split("-")[0],
                    "country": language.split("-")[1],
                }
            } if language else None,
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
        taxes: List[Taxes] = from_jsonarray_to_list(
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
        standardized_response = ReceiptParserDataClass(extracted_data=[ocr_receipt])

        result = ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
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
        # Invoice std : 
        invoice_data = original_response["document"]["inference"]["prediction"]
        default_dict = defaultdict(lambda: None)
        
        # Customer informations 
        customer_name = invoice_data.get("customer", default_dict).get("value", None)
        customer_address = invoice_data.get("customer_address", default_dict).get("value", None)
        
        # Merchant information
        merchant_name = invoice_data.get("supplier", default_dict).get("value", None)
        merchant_address = invoice_data.get("supplier_address", default_dict).get("value", None)
        
        # Others
        date = invoice_data.get("date", default_dict).get("value", None)
        time = invoice_data.get("time", default_dict).get("value", None)
        date = combine_date_with_time(date, time)
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
            merchant_information=MerchantInformationInvoice(
            merchant_name=merchant_name, merchant_address=merchant_address
            ),
            customer_information = CustomerInformationInvoice(
            customer_name=customer_name, customer_address=customer_address,
            customer_mailing_address = customer_address
            ),
            invoice_number=invoice_number,
            invoice_total=invoice_total,
            invoice_subtotal=invoice_subtotal,
            date=date,
            due_date=due_date,
            taxes=taxes,
            locale=LocaleInvoice(currency=currency, language=language),
        )

        standardized_response = InvoiceParserDataClass(
            extracted_data=[invoice_parser]
        ).dict()

        result = ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__identity_parser(self, file: BufferedReader) -> ResponseType[IdentityParserDataClass]:
        args = self._get_api_attributes(file)

        response = requests.post(url=self.url_identity, files=args['files'], headers=args['headers'])

        original_response = response.json()

        if response.status_code != 201:
            raise ProviderException(message=original_response['error']['message'], code=response.status_code)

        identity_data = original_response["document"]["inference"]["prediction"]

        given_names: Sequence[StrictStr] = []

        for given_name in identity_data['given_names']:
            given_names.append(ItemIdentityParserDataClass(
                value=given_name['value'],
                confidence=given_name['confidence'])
            )

        last_name = ItemIdentityParserDataClass(
            value=identity_data['surname']['value'],
            confidence=identity_data['surname']['confidence']
        )
        birth_date = ItemIdentityParserDataClass(
            value=identity_data['birth_date']['value'],
            confidence=identity_data['birth_date']['confidence']
        )
        birth_place = ItemIdentityParserDataClass(
            value=identity_data['birth_place']['value'],
            confidence=identity_data['birth_place']['confidence']
        )
        country = get_info_country(key=InfoCountry.ALPHA3, value=identity_data['country']['value'])
        if country:
            country['confidence'] = identity_data['country']['confidence']
        issuance_date = ItemIdentityParserDataClass(
            value=identity_data['issuance_date']['value'],
            confidence=identity_data['issuance_date']['confidence']
        )
        expire_date = ItemIdentityParserDataClass(
            value=identity_data['expiry_date']['value'],
            confidence=identity_data['expiry_date']['confidence']
        )
        document_id = ItemIdentityParserDataClass(
            value=identity_data['id_number']['value'],
            confidence=identity_data['id_number']['confidence']
        )
        gender = ItemIdentityParserDataClass(
            value=identity_data['gender']['value'],
            confidence=identity_data['gender']['confidence']
        )
        mrz = ItemIdentityParserDataClass(
            value=identity_data['mrz1']['value'],
            confidence=identity_data['mrz1']['confidence']
        )
        items: Sequence[InfosIdentityParserDataClass] = []
        items.append(InfosIdentityParserDataClass(
            last_name=last_name,
            given_names=given_names,
            birth_date=birth_date,
            birth_place=birth_place,
            country=country,
            issuance_date=issuance_date,
            expire_date=expire_date,
            document_id=document_id,
            gender=gender,
            mrz=mrz,
        ))

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
