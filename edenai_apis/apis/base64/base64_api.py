from itertools import zip_longest
import json
from typing import Dict, Sequence, TypeVar, Union
from collections import defaultdict
import mimetypes
import base64
from enum import Enum
import requests
from edenai_apis.features.ocr.identity_parser import (
    IdentityParserDataClass,
    InfoCountry,
    ItemIdentityParserDataClass,
    get_info_country,
    InfosIdentityParserDataClass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    format_date,
)
from edenai_apis.features.ocr.invoice_parser import (
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
    BankInvoice,
)
from edenai_apis.features.ocr.receipt_parser import (
    CustomerInformation,
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
    ReceiptParserDataClass,
    Taxes,
    PaymentInformation,
)
from edenai_apis.features.image.face_compare import (
    FaceCompareDataClass,
    FaceMatch,
    FaceCompareBoundingBox,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
    retreive_first_number_from_string,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class SubfeatureParser(Enum):
    RECEIPT = "receipt"
    INVOICE = "invoice"


T = TypeVar("T")


class Base64Api(ProviderInterface, OcrInterface):
    provider_name = "base64"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["secret"]
        self.url = "https://base64.ai/api/scan"

    def _extract_item_lignes(
        self, data, item_lines_type: Union[ItemLines, ItemLinesInvoice]
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
                    quantity=convert_string_to_number(item_quantity, int),
                    amount=convert_string_to_number(item[2], float),
                    unit_price=convert_string_to_number(item[3], float),
                )
            )
        return items

    def _format_invoice_document_data(self, data) -> InvoiceParserDataClass:
        fields = data[0].get("fields", [])

        items: Sequence[ItemLinesInvoice] = self._extract_item_lignes(
            fields, ItemLinesInvoice
        )

        default_dict = defaultdict(lambda: None)
        # ----------------------Merchant & customer informations----------------------#
        merchant_name = fields.get("companyName", default_dict)["value"]
        merchant_address = fields.get("from", default_dict)["value"]
        customer_name = fields.get("billTo", default_dict)["value"]
        customer_address = fields.get("address", default_dict)[
            "value"
        ]  # Deprecated need to be removed
        customer_mailing_address = fields.get("address", default_dict)["value"]
        customer_billing_address = fields.get("billTo", default_dict)["value"]
        customer_shipping_address = fields.get("shipTo", default_dict)["value"]
        customer_remittance_address = fields.get("soldTo", default_dict)["value"]
        # ---------------------- invoice  informations----------------------#
        invoice_number = fields.get("invoiceNumber", default_dict)["value"]
        invoice_total = fields.get("total", default_dict)["value"]
        invoice_total = convert_string_to_number(invoice_total, float)
        invoice_subtotal = fields.get("subtotal", default_dict)["value"]
        invoice_subtotal = convert_string_to_number(invoice_subtotal, float)
        amount_due = fields.get("balanceDue", default_dict)["value"]
        amount_due = convert_string_to_number(amount_due, float)
        discount = fields.get("discount", default_dict)["value"]
        discount = convert_string_to_number(discount, float)
        taxe = fields.get("tax", default_dict)["value"]
        taxe = convert_string_to_number(taxe, float)
        taxes: Sequence[TaxesInvoice] = [(TaxesInvoice(value=taxe))]
        # ---------------------- payment informations----------------------#
        payment_term = fields.get("paymentTerms", default_dict)["value"]
        purchase_order = fields.get("purchaseOrder", default_dict)["value"]
        date = fields.get("invoiceDate", default_dict)["value"]
        time = fields.get("invoiceTime", default_dict)["value"]
        date = combine_date_with_time(date, time)
        due_date = fields.get("dueDate", default_dict)["value"]
        due_time = fields.get("dueTime", default_dict)["value"]
        due_date = combine_date_with_time(due_date, due_time)
        # ---------------------- bank and local informations----------------------#
        iban = fields.get("iban", default_dict)["value"]
        account_number = fields.get("accountNumber", default_dict)["value"]
        currency = fields.get("currency", default_dict)["value"]

        invoice_parser = InfosInvoiceParserDataClass(
            merchant_information=MerchantInformationInvoice(
                merchant_name=merchant_name, merchant_address=merchant_address
            ),
            customer_information=CustomerInformationInvoice(
                customer_name=customer_name,
                customer_address=customer_address,
                customer_mailing_address=customer_mailing_address,
                customer_remittance_address=customer_remittance_address,
                customer_shipping_address=customer_shipping_address,
                customer_billing_address=customer_billing_address,
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
            locale=LocaleInvoice(currency=currency),
            bank_informations=BankInvoice(iban=iban, account_number=account_number),
            item_lines=items,
        )

        standardized_response = InvoiceParserDataClass(extracted_data=[invoice_parser])

        return standardized_response

    def _format_receipt_document_data(self, data) -> ReceiptParserDataClass:
        fields = data[0].get("fields", [])

        items: Sequence[ItemLines] = self._extract_item_lignes(fields, ItemLines)

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
        taxes: Sequence[Taxes] = [(Taxes(taxes=taxe))]
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

    def _send_ocr_document(self, file: str, model_type: str) -> Dict:
        file_ = open(file, "rb")
        image_as_base64 = (
            f"data:{mimetypes.guess_type(file)[0]};base64,"
            + base64.b64encode(file_.read()).decode()
        )
        file_.close()

        data = {"modelTypes": [model_type], "image": image_as_base64}

        headers = {"Content-type": "application/json", "Authorization": self.api_key}

        response = requests.post(url=self.url, headers=headers, json=data)

        if response.status_code != 200:
            raise ProviderException(response.text)

        return response.json()

    def _ocr_finance_document(
        self, ocr_file, document_type: SubfeatureParser
    ) -> ResponseType[T]:
        original_response = self._send_ocr_document(
            ocr_file, "finance/" + document_type.value
        )
        if document_type == SubfeatureParser.RECEIPT:
            standardized_response = self._format_receipt_document_data(
                original_response
            )
        elif document_type == SubfeatureParser.INVOICE:
            standardized_response = self._format_invoice_document_data(
                original_response
            )

        result = ResponseType[T](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__ocr(
        self,
        file: str,
        language: str,
        file_url: str = "",
    ):
        raise ProviderException(
            message="This provider is deprecated. You won't be charged for your call."
        )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[InvoiceParserDataClass]:
        return self._ocr_finance_document(file, SubfeatureParser.INVOICE)

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[ReceiptParserDataClass]:
        return self._ocr_finance_document(file, SubfeatureParser.RECEIPT)

    def ocr__identity_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[IdentityParserDataClass]:
        file_ = open(file, "rb")

        image_as_base64 = (
            f"data:{mimetypes.guess_type(file)[0]};base64,"
            + base64.b64encode(file_.read()).decode()
        )

        payload = json.dumps({"image": image_as_base64})

        headers = {"Content-Type": "application/json", "Authorization": self.api_key}

        response = requests.post(url=self.url, headers=headers, data=payload)

        file_.close()

        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(message=original_response["message"])

        items = []

        for document in original_response:
            image_id = [
                ItemIdentityParserDataClass(
                    value=doc.get("image", []), confidence=doc.get("confidence")
                )
                for doc in document["features"].get("faces", {})
            ]
            image_signature = [
                ItemIdentityParserDataClass(
                    value=doc.get("image", []), confidence=doc.get("confidence")
                )
                for doc in document["features"].get("signatures", {})
            ]
            given_names = (
                document["fields"].get("givenName", {}).get("value", "").split(" ")
                if document["fields"].get("givenName", {}).get("value", "") != ""
                else []
            )
            given_names_final = []
            for given_name in given_names:
                given_names_final.append(
                    ItemIdentityParserDataClass(
                        value=given_name,
                        confidence=document["fields"]
                        .get("givenName", {})
                        .get("confidence"),
                    )
                )

            country = get_info_country(
                key=InfoCountry.ALPHA3,
                value=document["fields"].get("countryCode", {}).get("value", ""),
            )
            if country:
                country["confidence"] = (
                    document["fields"].get("countryCode", {}).get("confidence")
                )

            items.append(
                InfosIdentityParserDataClass(
                    document_type=ItemIdentityParserDataClass(
                        value=document["fields"].get("documentType", {}).get("value"),
                        confidence=document["fields"]
                        .get("documentType", {})
                        .get("confidence"),
                    ),
                    last_name=ItemIdentityParserDataClass(
                        value=document["fields"].get("familyName", {}).get("value"),
                        confidence=document["fields"]
                        .get("familyName", {})
                        .get("confidence"),
                    ),
                    given_names=given_names_final,
                    birth_date=ItemIdentityParserDataClass(
                        value=format_date(
                            document["fields"].get("dateOfBirth", {}).get("value")
                        ),
                        confidence=document["fields"]
                        .get("dateOfBirth", {})
                        .get("confidence"),
                    ),
                    country=country,
                    document_id=ItemIdentityParserDataClass(
                        value=document["fields"].get("documentNumber", {}).get("value"),
                        confidence=document["fields"]
                        .get("documentNumber", {})
                        .get("confidence"),
                    ),
                    age=ItemIdentityParserDataClass(
                        value=str(document["fields"].get("age", {}).get("value")),
                        confidence=document["fields"].get("age", {}).get("confidence"),
                    ),
                    nationality=ItemIdentityParserDataClass(
                        value=document["fields"].get("nationality", {}).get("value"),
                        confidence=document["fields"]
                        .get("nationality", {})
                        .get("confidence"),
                    ),
                    issuing_state=ItemIdentityParserDataClass(
                        value=document["fields"].get("issuingState", {}).get("value"),
                        confidence=document["fields"]
                        .get("issuingState", {})
                        .get("confidence"),
                    ),
                    image_id=image_id,
                    image_signature=image_signature,
                    gender=ItemIdentityParserDataClass(
                        value=document["fields"].get("sex", {}).get("value"),
                        confidence=document["fields"].get("sex", {}).get("confidence"),
                    ),
                    expire_date=ItemIdentityParserDataClass(
                        value=format_date(
                            document["fields"].get("expirationDate", {}).get("value")
                        ),
                        confidence=document["fields"]
                        .get("expirationDate", {})
                        .get("confidence"),
                    ),
                    issuance_date=ItemIdentityParserDataClass(
                        value=format_date(
                            document["fields"].get("issueDate", {}).get("value")
                        ),
                        confidence=document["fields"]
                        .get("issueDate", {})
                        .get("confidence"),
                    ),
                    address=ItemIdentityParserDataClass(
                        value=document["fields"].get("address", {}).get("value"),
                        confidence=document["fields"]
                        .get("address", {})
                        .get("confidence"),
                    ),
                )
            )

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__face_compare(
        self,
        file1: str,
        file2: str,
        file1_url: str = "",
        file2_url: str = "",
        ) -> ResponseType[FaceCompareDataClass]:
        url = "https://base64.ai/api/face"
        
        headers = {
        'Authorization': self.api_key,
        'Content-Type': 'application/json'
        }
        
        if file1_url and file2_url:
            payload = json.dumps({
                "referenceUrl": file1_url,
                "queryUrl": file2_url
            })
        else:
            file_reference_ = open(file1, "rb")
            file_query_ = open(file2, "rb")
            image_reference_as_base64 = (
                f"data:{mimetypes.guess_type(file1)[0]};base64,"
                + base64.b64encode(file_reference_.read()).decode()
            )
            image_query_as_base64 = (
                f"data:{mimetypes.guess_type(file2)[0]};base64,"
                + base64.b64encode(file_query_.read()).decode()
            )
            payload = json.dumps({
                "referenceImage": image_reference_as_base64,
                "queryImage": image_query_as_base64
            })
        
        response = requests.request("POST", url, headers=headers, data=payload)
        original_response = response.json()
        
        if response.status_code != 200:
            raise ProviderException(message=original_response['message'])
        
        faces = []
        for matching_face in original_response.get('matches',[]):
            faces.append(
                FaceMatch(
                    confidence=matching_face.get('confidence'),
                    bounding_box=FaceCompareBoundingBox(
                        top=matching_face.get('top'),
                        left=matching_face.get('left'),
                        height=matching_face.get('height'),
                        width=matching_face.get('width'),
                    )
                )
            )
        standardized_response = FaceCompareDataClass(items=faces)

        return ResponseType[FaceCompareDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )