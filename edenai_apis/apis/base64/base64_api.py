import base64
import json
import mimetypes
import uuid
from enum import Enum
from io import BytesIO
from typing import Any, Dict, Sequence, Type, TypeVar, Union

import requests

from edenai_apis.apis.base64.base64_helpers import (
    extract_item_lignes,
    format_financial_document_data,
    format_invoice_document_data,
    format_receipt_document_data,
)
from edenai_apis.features import OcrInterface, ProviderInterface
from edenai_apis.features.image.face_compare import (
    FaceCompareBoundingBox,
    FaceCompareDataClass,
    FaceMatch,
)
from edenai_apis.features.ocr.anonymization_async.anonymization_async_dataclass import (
    AnonymizationAsyncDataClass,
)
from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    MicrModel,
)
from edenai_apis.features.ocr.bank_check_parsing.bank_check_parsing_dataclass import (
    ItemBankCheckParsingDataClass,
)
from edenai_apis.features.ocr.data_extraction.data_extraction_dataclass import (
    DataExtractionDataClass,
    ItemDataExtraction,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.features.ocr.identity_parser import (
    IdentityParserDataClass,
    InfoCountry,
    InfosIdentityParserDataClass,
    ItemIdentityParserDataClass,
    get_info_country,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    Country,
    format_date,
)
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.ocr.ocr_dataclass import OcrDataClass
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.bounding_box import BoundingBox
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


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

    class Field:
        def __init__(self, document: dict) -> None:
            self.document = document

        def __getitem__(self, key) -> Any:
            return self.document.get("fields", {}).get(key, {}).get("value")

    def _get_response(self, response: requests.Response) -> Any:
        try:
            original_response = response.json()
            if response.status_code >= 400:
                message_error = original_response["message"]
                raise ProviderException(message_error, code=response.status_code)
            return original_response
        except Exception:
            raise ProviderException(response.text, code=response.status_code)

    def _send_ocr_document(self, file: str, model_type: str) -> Dict:
        with open(file, "rb") as file_:
            image_as_base64 = (
                f"data:{mimetypes.guess_type(file)[0]};base64,"
                + base64.b64encode(file_.read()).decode()
            )

        data = {"modelTypes": [model_type], "image": image_as_base64}

        headers = {"Content-type": "application/json", "Authorization": self.api_key}

        response = requests.post(url=self.url, headers=headers, json=data)

        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        return response.json()

    def _ocr_finance_document(
        self, ocr_file, document_type: SubfeatureParser
    ) -> ResponseType[T]:
        original_response = self._send_ocr_document(
            ocr_file, "finance/" + document_type.value
        )
        if document_type == SubfeatureParser.RECEIPT:
            standardized_response = format_receipt_document_data(original_response)
        elif document_type == SubfeatureParser.INVOICE:
            standardized_response = format_invoice_document_data(original_response)

        result = ResponseType[T](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        raise ProviderException(
            message="This provider is deprecated. You won't be charged for your call.",
            code=500,
        )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        return self._ocr_finance_document(file, SubfeatureParser.INVOICE)

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        return self._ocr_finance_document(file, SubfeatureParser.RECEIPT)

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        original_response = self._send_ocr_document(file, "finance/" + document_type)

        standardized_response = format_financial_document_data(original_response)
        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:
        with open(file, "rb") as file_:
            image_as_base64 = (
                f"data:{mimetypes.guess_type(file)[0]};base64,"
                + base64.b64encode(file_.read()).decode()
            )

            payload = json.dumps({"image": image_as_base64})

            headers = {
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }

            response = requests.post(url=self.url, headers=headers, data=payload)

        original_response = self._get_response(response)

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
            given_names_dict = document["fields"].get("givenName", {}) or {}
            given_names_string = given_names_dict.get("value", "") or ""
            given_names = (
                given_names_string.split(" ") if given_names_string != "" else []
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
                    country=country or Country.default(),
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
                    birth_place=ItemIdentityParserDataClass(
                        value=None, confidence=None
                    ),
                    mrz=ItemIdentityParserDataClass(),
                )
            )

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__face_compare(
        self, file1: str, file2: str, file1_url: str = "", file2_url: str = "", **kwargs
    ) -> ResponseType[FaceCompareDataClass]:
        url = "https://base64.ai/api/face"

        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}

        if file1_url and file2_url:
            payload = json.dumps({"url": file1_url, "queryUrl": file2_url})
        else:
            with open(file1, "rb") as file_reference_, open(file2, "rb") as file_query_:

                image_reference_as_base64 = (
                    f"data:{mimetypes.guess_type(file1)[0]};base64,"
                    + base64.b64encode(file_reference_.read()).decode()
                )
                image_query_as_base64 = (
                    f"data:{mimetypes.guess_type(file2)[0]};base64,"
                    + base64.b64encode(file_query_.read()).decode()
                )
                payload = json.dumps(
                    {
                        "document": image_reference_as_base64,
                        "query": image_query_as_base64,
                    }
                )

        response = requests.request("POST", url, headers=headers, data=payload)
        original_response = self._get_response(response)

        faces = []
        for matching_face in original_response.get("matches", []):
            faces.append(
                FaceMatch(
                    confidence=matching_face.get("confidence") or 0,
                    bounding_box=FaceCompareBoundingBox(
                        top=matching_face.get("top"),
                        left=matching_face.get("left"),
                        height=matching_face.get("height"),
                        width=matching_face.get("width"),
                    ),
                )
            )
        standardized_response = FaceCompareDataClass(items=faces)

        return ResponseType[FaceCompareDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__data_extraction(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[DataExtractionDataClass]:
        with open(file, "rb") as f_stream:
            image_as_base64 = (
                f"data:{mimetypes.guess_type(file)[0]};base64,"
                + base64.b64encode(f_stream.read()).decode()
            )

            payload = json.dumps({"image": image_as_base64})
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }

            response = requests.post(url=self.url, headers=headers, data=payload)

        original_response = self._get_response(response)

        items: Sequence[ItemDataExtraction] = []

        for document in original_response:
            for _, value in document.get("fields", {}).items():
                try:
                    bbox = BoundingBox.from_normalized_vertices(
                        normalized_vertices=value.get("location")
                    )
                except ValueError:
                    bbox = BoundingBox.unknown()

                if key := value.get("key"):
                    items.append(
                        ItemDataExtraction(
                            key=key,
                            value=value.get("value"),
                            confidence_score=value.get("confidence"),
                            bounding_box=bbox,
                        )
                    )

        standardized_response = DataExtractionDataClass(fields=items)

        return ResponseType(
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__bank_check_parsing(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[BankCheckParsingDataClass]:
        with open(file, "rb") as fstream:
            image_as_base64 = (
                f"data:{mimetypes.guess_type(file)[0]};base64,"
                + base64.b64encode(fstream.read()).decode()
            )

            payload = json.dumps({"modelTypes": ["finance/"], "image": image_as_base64})
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.api_key,
            }

            response = requests.post(url=self.url, headers=headers, data=payload)
            original_response = self._get_response(response)

            items: Sequence[ItemBankCheckParsingDataClass] = []
            for fields_not_formated in original_response:
                fields = Base64Api.Field(fields_not_formated)
                items.append(
                    ItemBankCheckParsingDataClass(
                        amount=fields["amount"],
                        amount_text=None,
                        bank_name=None,
                        bank_address=None,
                        date=fields["date"],
                        memo=None,
                        payer_address=fields["address"],
                        payer_name=fields["payee"],
                        receiver_name=None,
                        receiver_address=None,
                        currency=fields["currency"],
                        micr=MicrModel(
                            raw=fields["micr"],
                            account_number=fields["accountMumber"],
                            serial_number=None,
                            check_number=fields["checkNumber"],
                            routing_number=fields["routingNumber"],
                        ),
                    )
                )
            return ResponseType[BankCheckParsingDataClass](
                original_response=original_response,
                standardized_response=BankCheckParsingDataClass(extracted_data=items),
            )

    def ocr__anonymization_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        data_job_id = {}
        with open(file, "rb") as file_:
            image_as_base64 = (
                f"data:{mimetypes.guess_type(file)[0]};base64,"
                + base64.b64encode(file_.read()).decode()
            )

        payload = json.dumps(
            {
                "image": image_as_base64,
                "settings": {
                    "redactions": {
                        "fields": [
                            "name",
                            "givenName",
                            "familyName",
                            "organization",
                            "documentNumber",
                            "address",
                            "date",
                            "dateOfBirth",
                            "issueDate",
                            "expirationDate",
                            "vin" "total",
                            "tax",
                        ],
                        "faces": True,
                        "signatures": True,
                    }
                },
            }
        )

        headers = {"Content-Type": "application/json", "Authorization": self.api_key}

        response = requests.post(url=self.url, headers=headers, data=payload)

        original_response = self._get_response(response)

        job_id = "document_anonymization_base64" + str(uuid.uuid4())
        # Extract the B64 redacted document
        redacted_document = original_response[0].get("redactedDocument")
        # document_mimetype = original_response[0]['features']['properties']['mimeType']

        # # Use the mimetypes module to guess the file extension based on the MIME type
        # extension = mimetypes.guess_extension(document_mimetype)

        # Extract the base64-encoded data from 'redacted_document'
        base64_data = redacted_document.split(";base64,")[1]

        content_bytes = base64.b64decode(base64_data)
        resource_url = upload_file_bytes_to_s3(
            BytesIO(content_bytes), ".png", USER_PROCESS
        )
        return AsyncResponseType[AnonymizationAsyncDataClass](
            original_response=original_response,
            standardized_response=AnonymizationAsyncDataClass(
                document=base64_data, document_url=resource_url
            ),
            provider_job_id=job_id,
        )
