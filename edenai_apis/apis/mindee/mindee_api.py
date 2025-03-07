import json
from collections import defaultdict
from io import BufferedReader
from typing import Dict, Optional, Sequence, TypeVar, TypedDict

import requests

from edenai_apis.apis.mindee.mindee_ocr_normalizer import mindee_financial_parser
from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr import (
    ReceiptParserDataClass,
    InfosInvoiceParserDataClass,
    InfosReceiptParserDataClass,
    InvoiceParserDataClass,
    IdentityParserDataClass,
    InfosIdentityParserDataClass,
    InfoCountry,
    get_info_country,
    ItemIdentityParserDataClass,
)
from edenai_apis.features.ocr.bank_check_parsing import (
    BankCheckParsingDataClass,
    MicrModel,
    ItemBankCheckParsingDataClass,
)
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import Country
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import (
    CustomerInformationInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
from edenai_apis.features.ocr.invoice_splitter_async.invoice_splitter_async_dataclass import (
    InvoiceSplitterAsyncDataClass,
    InvoiceGroupDataClass,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    CustomerInformation,
    ItemLines,
    Locale,
    MerchantInformation,
    PaymentInformation,
    Taxes,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    ResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncErrorResponseType,
    AsyncBaseResponseType,
    AsyncResponseType,
)


class RequestParams(TypedDict):
    headers: Dict[str, str]
    files: Dict[str, BufferedReader]


class MindeeApi(ProviderInterface, OcrInterface):
    provider_name = "mindee"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["subscription_key"]
        self.url = "https://api.mindee.net/v1/products/mindee/invoices/v3/predict"
        self.url_receipt = (
            "https://api.mindee.net/v1/products/mindee/expense_receipts/v5/predict"
        )
        self.url_identity = (
            "https://api.mindee.net/v1/products/mindee/passport/v1/predict"
        )
        self.url_financial = (
            "https://api.mindee.net/v1/products/mindee/financial_document/v1/predict"
        )
        self.url_bank_check = (
            "https://api.mindee.net/v1/products/mindee/bank_check/v1/predict"
        )
        self.url_invoice_splitter = (
            "https://api.mindee.net/v1/products/mindee/invoice_splitter/v1/"
        )

    def _get_api_attributes(self, file: BufferedReader) -> RequestParams:
        return RequestParams(
            headers={"Authorization": self.api_key},
            files={"document": file},
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        with open(file, "rb") as file_:
            args = self._get_api_attributes(file_)
            response = requests.post(
                self.url_receipt,
                headers=args["headers"],
                files=args["files"],
            )
            original_response = response.json()

        if "document" not in original_response:
            raise ProviderException(
                original_response["api_request"]["error"]["message"],
                code=response.status_code,
            )

        extracted_data = []
        for page in original_response["document"]["inference"]["pages"]:
            receipt_data = page["prediction"]
            supplier_company_registrations = receipt_data.get(
                "supplier_company_registrations", None
            )
            merchant_siret = None
            merchant_siren = None
            if supplier_company_registrations:
                for supplier_info in supplier_company_registrations:
                    supplier_type = supplier_info.get("type", None)
                    if supplier_type:
                        if supplier_type == "SIRET":
                            merchant_siret = supplier_info.get("value", None)
                        if supplier_type == "SIREN":
                            merchant_siren = supplier_info.get("value", None)
            extracted_data.append(
                InfosReceiptParserDataClass(
                    invoice_number=None,
                    invoice_total=receipt_data["total_amount"]["value"],
                    invoice_subtotal=None,
                    barcodes=[],
                    date=combine_date_with_time(
                        receipt_data["date"]["value"], receipt_data["time"]["value"]
                    ),
                    due_date=None,
                    customer_information=CustomerInformation(customer_name=None),
                    merchant_information=MerchantInformation(
                        merchant_name=receipt_data["supplier_name"]["value"],
                        merchant_address=receipt_data["supplier_address"]["value"],
                        merchant_phone=receipt_data["supplier_phone_number"]["value"],
                        merchant_url=None,
                        merchant_siret=merchant_siret,
                        merchant_siren=merchant_siren,
                    ),
                    payment_information=PaymentInformation(
                        card_number=None,
                        card_type=None,
                        cash=None,
                        tip=None,
                        change=None,
                        discount=None,
                    ),
                    locale=Locale(
                        currency=receipt_data["locale"]["currency"],
                        language=receipt_data["locale"]["language"],
                        country=receipt_data["locale"]["country"],
                    ),
                    taxes=[
                        Taxes(
                            taxes=tax["value"],
                            rate=tax["rate"],
                        )
                        for tax in receipt_data["taxes"]
                    ],
                    item_lines=[
                        ItemLines(
                            description=item["description"],
                            quantity=item["quantity"],
                            unit_price=item["unit_price"],
                            amount=item["total_amount"],
                        )
                        for item in receipt_data["line_items"]
                    ],
                )
            )

        standardized_response = ReceiptParserDataClass(extracted_data=extracted_data)

        result = ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        headers = {
            "Authorization": self.api_key,
        }
        with open(file, "rb") as file_:
            files = {"document": file_}
            response = requests.post(self.url, headers=headers, files=files)
            original_response = response.json()

        if "document" not in original_response:
            raise ProviderException(
                original_response["api_request"]["error"]["message"], code=response
            )
        # Invoice std :
        extracted_data = []
        for page in original_response["document"]["inference"]["pages"]:
            invoice_data = page["prediction"]

            default_dict = defaultdict(lambda: None)

            # Customer informations
            customer_name = invoice_data.get("customer", default_dict).get(
                "value", None
            )
            customer_address = invoice_data.get("customer_address", default_dict).get(
                "value", None
            )
            customer_company_registrations = invoice_data.get(
                "customer_company_registrations", {}
            )
            customer_vat_number = None
            customer_abn_number = None
            customer_gst_number = None
            customer_siret_number = None
            customer_siren_number = None
            for customer_info in customer_company_registrations:
                customer_type = customer_info.get("type", "")
                customer_registration_value = customer_info.get("value", "")
                if customer_type == "VAT NUMBER":
                    customer_vat_number = customer_registration_value
                elif customer_type == "ABN":
                    customer_abn_number = customer_registration_value
                elif customer_type == "GSTIN":
                    customer_gst_number = customer_registration_value
                elif customer_type == "SIREN":
                    customer_siren_number = customer_registration_value
                elif customer_type == "SIRET":
                    customer_siret_number = customer_registration_value

            # Merchant information
            merchant_name = invoice_data.get("supplier", default_dict).get(
                "value", None
            )
            merchant_address = invoice_data.get("supplier_address", default_dict).get(
                "value", None
            )
            supplier_company_registrations = invoice_data.get(
                "supplier_company_registrations", {}
            )
            merchant_siret = None
            merchant_siren = None
            merchant_vat_number = None
            merchant_abn_number = None
            merchant_gst_number = None
            for supplier_info in supplier_company_registrations:
                supplier_type = supplier_info.get("type", "")
                supplier_registration_value = supplier_info.get("value", None)
                if supplier_type == "SIRET":
                    merchant_siret = supplier_registration_value
                elif supplier_type == "SIREN":
                    merchant_siren = supplier_registration_value
                elif supplier_type == "VAT NUMBER":
                    merchant_vat_number = supplier_registration_value
                elif supplier_type == "ABN":
                    merchant_abn_number = supplier_registration_value
                elif supplier_type == "GSTIN":
                    merchant_gst_number = supplier_registration_value
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

            extracted_data.append(
                InfosInvoiceParserDataClass(
                    merchant_information=MerchantInformationInvoice(
                        merchant_name=merchant_name,
                        merchant_address=merchant_address,
                        # Not supported by the Mindee
                        # --------------------------------
                        merchant_phone=None,
                        merchant_email=None,
                        merchant_fax=None,
                        merchant_website=None,
                        merchant_siret=merchant_siret,
                        merchant_siren=merchant_siren,
                        merchant_tax_id=None,
                        abn_number=merchant_abn_number,
                        gst_number=merchant_gst_number,
                        vat_number=merchant_vat_number,
                        pan_number=None,
                        # --------------------------------
                    ),
                    customer_information=CustomerInformationInvoice(
                        customer_name=customer_name,
                        customer_address=customer_address,
                        customer_mailing_address=customer_address,
                        customer_email=None,
                        customer_id=None,
                        customer_tax_id=None,
                        customer_billing_address=None,
                        customer_remittance_address=None,
                        customer_service_address=None,
                        customer_shipping_address=None,
                        abn_number=customer_abn_number,
                        gst_number=customer_gst_number,
                        pan_number=None,
                        vat_number=customer_vat_number,
                        siren_number=customer_siren_number,
                        siret_number=customer_siret_number,
                    ),
                    invoice_number=invoice_number,
                    invoice_total=invoice_total,
                    invoice_subtotal=invoice_subtotal,
                    date=date,
                    due_date=due_date,
                    taxes=taxes,
                    locale=LocaleInvoice(currency=currency, language=language),
                )
            )

        standardized_response = InvoiceParserDataClass(extracted_data=extracted_data)

        result = ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:
        with open(file, "rb") as file_:
            args = self._get_api_attributes(file_)

            response = requests.post(
                url=self.url_identity, files=args["files"], headers=args["headers"]
            )

        original_response = response.json()
        if response.status_code != 201:
            err_title = original_response["api_request"]["error"]["message"]
            err_msg = original_response["api_request"]["error"]["details"]
            raise ProviderException(
                message=f"{err_title}: {err_msg}",
                code=response.status_code,
            )

        identity_data = original_response["document"]["inference"]["prediction"]

        given_names: Sequence[ItemIdentityParserDataClass] = []

        for given_name in identity_data["given_names"]:
            given_names.append(
                ItemIdentityParserDataClass(
                    value=given_name["value"], confidence=given_name["confidence"]
                )
            )

        last_name = ItemIdentityParserDataClass(
            value=identity_data["surname"]["value"],
            confidence=identity_data["surname"]["confidence"],
        )
        birth_date = ItemIdentityParserDataClass(
            value=identity_data["birth_date"]["value"],
            confidence=identity_data["birth_date"]["confidence"],
        )
        birth_place = ItemIdentityParserDataClass(
            value=identity_data["birth_place"]["value"],
            confidence=identity_data["birth_place"]["confidence"],
        )

        country: Country = get_info_country(
            key=InfoCountry.ALPHA3, value=identity_data["country"]["value"]
        )
        if country:
            country["confidence"] = identity_data["country"]["confidence"]

        issuance_date = ItemIdentityParserDataClass(
            value=identity_data["issuance_date"]["value"],
            confidence=identity_data["issuance_date"]["confidence"],
        )
        expire_date = ItemIdentityParserDataClass(
            value=identity_data["expiry_date"]["value"],
            confidence=identity_data["expiry_date"]["confidence"],
        )
        document_id = ItemIdentityParserDataClass(
            value=identity_data["id_number"]["value"],
            confidence=identity_data["id_number"]["confidence"],
        )
        gender = ItemIdentityParserDataClass(
            value=identity_data["gender"]["value"],
            confidence=identity_data["gender"]["confidence"],
        )
        mrz = ItemIdentityParserDataClass(
            value=identity_data["mrz1"]["value"],
            confidence=identity_data["mrz1"]["confidence"],
        )
        items: Sequence[InfosIdentityParserDataClass] = []
        items.append(
            InfosIdentityParserDataClass(
                last_name=last_name,
                given_names=given_names,
                birth_date=birth_date,
                birth_place=birth_place,
                country=country or Country.default(),
                issuance_date=issuance_date,
                expire_date=expire_date,
                document_id=document_id,
                gender=gender,
                mrz=mrz,
                image_id=[],
                issuing_state=ItemIdentityParserDataClass(),
                address=ItemIdentityParserDataClass(),
                age=ItemIdentityParserDataClass(),
                document_type=ItemIdentityParserDataClass(),
                nationality=ItemIdentityParserDataClass(),
                image_signature=[],
            )
        )

        standardized_response = IdentityParserDataClass(extracted_data=items)

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__bank_check_parsing(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[BankCheckParsingDataClass]:
        with open(file, "rb") as file_:
            headers = {
                "Authorization": self.api_key,
            }
            files = {"document": file_}

            try:
                response = requests.post(
                    self.url_bank_check, headers=headers, files=files
                )
            except:
                raise ProviderException(
                    "Something went wrong when calling this feature", code=500
                )
            original_response = response.json()
            if response.status_code >= 400 or "document" not in original_response:
                api_response = original_response.get("api_request", {}) or {}
                error = api_response.get("error", {}) or {}
                error_message = (
                    error.get("message", "")
                    or "A provider error occurred while calling this feature"
                )
                raise ProviderException(error_message, code=response.status_code)
        bank_check_data = original_response["document"]["inference"]["prediction"]
        default_dict = defaultdict(lambda: None)

        payees_list = bank_check_data.get("payees", []) or []
        payees_list_value = []
        for p in payees_list:
            if p:
                payees_list_value.append(p.get("value") or "")
        payees_str = None
        if len(payees_list_value) > 0:
            payees_str = ",".join(payees_list_value)

        extracted_data = [
            ItemBankCheckParsingDataClass(
                amount=bank_check_data.get("amount", default_dict).get("value", None),
                amount_text=None,
                bank_address=None,
                bank_name=None,
                date=bank_check_data.get("date", default_dict).get("value", None),
                memo=None,
                payer_address=None,
                payer_name=payees_str,
                receiver_address=None,
                receiver_name=None,
                currency=None,
                micr=MicrModel(
                    raw=None,
                    account_number=bank_check_data.get(
                        "account_number", default_dict
                    ).get("value", None),
                    routing_number=bank_check_data.get(
                        "routing_number", default_dict
                    ).get("value", None),
                    serial_number=None,
                    check_number=bank_check_data.get("check_number", default_dict).get(
                        "value", None
                    ),
                ),
            )
        ]
        return ResponseType[BankCheckParsingDataClass](
            original_response=original_response,
            standardized_response=BankCheckParsingDataClass(
                extracted_data=extracted_data
            ),
        )

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        headers = {
            "Authorization": self.api_key,
        }
        with open(file, "rb") as file_:
            files = {"document": file_}
            response = requests.post(self.url_financial, headers=headers, files=files)
            original_response = response.json()

        if "document" not in original_response:
            raise ProviderException(
                original_response["api_request"]["error"]["message"], code=response
            )
        standardized_response = mindee_financial_parser(
            original_response=original_response
        )
        return ResponseType[FinancialParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def ocr__invoice_splitter_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        with open(file, "rb") as file_:
            args = self._get_api_attributes(file_)
            response = requests.post(
                url=self.url_invoice_splitter + "predict_async",
                headers=args["headers"],
                files=args["files"],
            )

            if response.status_code != 202:
                raise ProviderException(
                    response.json()["api_request"]["error"],
                    code=response.status_code,
                )
        return AsyncLaunchJobResponseType(provider_job_id=response.json()["job"]["id"])

    def ocr__invoice_splitter_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[InvoiceSplitterAsyncDataClass]:
        headers = {
            "Authorization": self.api_key,
        }

        response = requests.get(
            f"{self.url_invoice_splitter}documents/queue/{provider_job_id}",
            headers=headers,
        )

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                f"Error while decoding the response: {exc}", code=response.status_code
            ) from exc

        if original_response["api_request"]["status"] == "failure":
            return AsyncErrorResponseType(
                provider_job_id=provider_job_id,
                error=original_response["api_request"]["error"],
            )

        if original_response["job"]["status"] in ["processing", "waiting"]:
            return AsyncPendingResponseType(provider_job_id=provider_job_id)

        extracted_data = []
        for group in original_response["document"]["inference"]["prediction"][
            "invoice_page_groups"
        ]:
            extracted_data.append(
                InvoiceGroupDataClass(
                    page_indexes=group["page_indexes"],
                    confidence=group["confidence"],
                )
            )

        return AsyncResponseType[InvoiceSplitterAsyncDataClass](
            provider_job_id=provider_job_id,
            status="succeeded",
            original_response=original_response,
            standardized_response=InvoiceSplitterAsyncDataClass(
                extracted_data=extracted_data
            ),
        )
