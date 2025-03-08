from enum import Enum
from io import BufferedReader
from time import sleep
from typing import Dict

import requests

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
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class RossumApi(ProviderInterface, OcrInterface):
    provider_name = "rossum"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.username = self.api_settings["username"]
        self.password = self.api_settings["password"]
        self.url = "https://elis.rossum.ai/api/v1/"
        self.queue_id = self.api_settings["queue_id"]

        self._login()

    def _login(self):
        """
        Login to the provider and store the token in token attribute

        Raises:
            ProviderException: If the status code is not 200
        """
        response = requests.post(
            url=self.url + "auth/login",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
        )

        try:
            response_json = response.json()
        except Exception as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(
                message=response_json.get("detail", "Error while getting token"),
                code=response.status_code,
            )

        self.token = response_json["key"]

    class EndpointType(Enum):
        LOGIN = "LOGIN"
        UPLOAD = "UPLOAD"
        DOWNLOAD = "DOWNLOAD"

    def _get_endpoint(self, endpoint_type: EndpointType) -> str:
        """
        Get the endpoint from the endpoint type

        Args:
            endpoint_type (EndpointType): An enum of endpoint type (LOGIN, UPLOAD, STATUS, DOWNLOAD)

        Returns:
            str: The endpoint
        """
        self.endpoints = {
            "LOGIN": "auth/login",
            "UPLOAD": f"queues/{self.queue_id}/upload",
            "DOWNLOAD": f"queues/{self.queue_id}/export",
        }

        return self.url + self.endpoints[endpoint_type.value]

    def _upload(self, file: BufferedReader) -> tuple:
        """
        Upload a file to the provider

        Args:
            file (BufferedReader): The file to upload

        Returns:
           tuple: The tuple with the documents and the annotation endpoint (documents, annotation_endpoint)

        Raises:
            ProviderException: If an error occurs while uploading the file (Status code != 201)
        """
        response = requests.post(
            url=self._get_endpoint(self.EndpointType.UPLOAD),
            files={"content": file},
            headers={
                "Authorization": f"Token {self.token}",
            },
        )

        try:
            response_json = response.json()
        except Exception as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if response.status_code != 201:
            raise ProviderException(
                message=response_json.get("detail", "Error while uploading file"),
                code=response.status_code,
            )

        return (response_json["document"], response_json["annotation"])

    def _get_status_and_id(self, annotation_endpoint: str) -> tuple:
        """
        Get the status of the file and this id

        Args:
            annotation_endpoint (str): The annotation endpoint of the file

        Returns:
            tuple: The tuple with the id and the status of the file (id, status) (ex: ('122', 'to_review')))

        Raises:
            ProviderException: If an error occurs while checking the status (Status code != 200)
        """
        response = requests.get(
            url=annotation_endpoint, headers={"Authorization": f"Token {self.token}"}
        )

        try:
            response_json = response.json()
        except Exception as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(
                message=response_json.get("detail", "Error while checking status"),
                code=response.status_code,
            )

        return (response_json["id"], response_json["status"])

    def _download_reviewing_data(self, id: str) -> dict:
        """
        Download the reviewing data

        Returns:
            ResponseType[dict]: The reviewing data

        Raises:
            ProviderException: If an error occurs while downloading the reviewing data (Status code != 200)
        """
        response = requests.get(
            url=self._get_endpoint(self.EndpointType.DOWNLOAD)
            + f"?status=to_review&format=json&id={id}",
            headers={"Authorization": f"Token {self.token}"},
        )

        try:
            response_json = response.json()
        except Exception as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if response.status_code != 200:
            raise ProviderException(
                message=response_json.get(
                    "detail", "Error while downloading reviewing data"
                ),
                code=response.status_code,
            )

        return response_json

    class CategoryType(Enum):
        BASIC = "basic_information"
        PAYMENT = "payment_information"
        VAT_AND_AMOUNT = "vat_and_amount"
        VENDOR_AND_CUSTOMER = "vendor_and_customer"
        OTHER = "other"
        LINE_ITEMS = "line_items"

        @classmethod
        def as_list(cls):
            return [category.value for category in cls]

    @staticmethod
    def _parse_tuple_value_in_data(data: dict, field_name: str) -> dict:
        """
        Parse the tuple value in the data

        Args:
            data (dict): The data to parse
            field_name (str): The field name to parse

        Returns:
            dict: The parsed data
        """
        new_data = []
        for item in data[field_name]:
            new_data.append(
                {entry["schema_id"]: entry["value"] for entry in item["children"]}
            )
        return new_data

    @staticmethod
    def _parse_response_for_specific_page(
        original_response: dict, page_number: int = 0
    ) -> dict:
        """
        Parse the response for a specific page. If the page is not specified, the first page will be parsed.
        The page number starts at 0. For example, if you want to parse the second page, you have to specify 1.

        Args:
            original_response (dict): The original response of the provider
            page (int): The page to parse

        Returns:
            dict: The parsed response for the specific page with all fields of CategoryType
        """
        if original_response["pagination"]["total_pages"] <= page_number:
            raise IndexError("The page number is out of range")

        response_parsed = {}
        page = original_response["results"][page_number]
        for idx, category_information in enumerate(page["content"]):
            current_category = RossumApi.CategoryType.as_list()[idx]
            data_as_dict = {
                entry["schema_id"]: (
                    entry["value"]
                    if entry["category"] == "datapoint"
                    else entry["children"]
                )
                for entry in category_information["children"]
            }
            if current_category == RossumApi.CategoryType.LINE_ITEMS.value:
                data_as_dict = RossumApi._parse_tuple_value_in_data(
                    data_as_dict, "line_items"
                )
            elif current_category == RossumApi.CategoryType.VAT_AND_AMOUNT.value:
                data_as_dict["tax_details"] = RossumApi._parse_tuple_value_in_data(
                    data_as_dict, "tax_details"
                )
            response_parsed[current_category] = data_as_dict

        return response_parsed

    def _invoice_standardization(
        self, original_response: dict
    ) -> InvoiceParserDataClass:
        """
        Standardize the response of the provider

        Args:
            original_response (dict): The original response of the provider

        Returns:
            InvoiceParserDataClass: The standardized response
        """
        extracted_data = []
        for page_number in range(original_response["pagination"]["total_pages"]):
            response_parsed = self._parse_response_for_specific_page(
                original_response, page_number
            )
            customer_information = CustomerInformationInvoice(
                customer_name=response_parsed["vendor_and_customer"]["recipient_name"],
                customer_address=response_parsed["vendor_and_customer"][
                    "recipient_address"
                ],
                customer_email=None,
                customer_id=response_parsed["basic_information"]["customer_id"],
                customer_tax_id=response_parsed["vendor_and_customer"][
                    "recipient_vat_id"
                ],
                customer_mailing_address=None,
                customer_billing_address=None,
                customer_shipping_address=response_parsed["vendor_and_customer"][
                    "recipient_delivery_address"
                ],
                customer_service_address=None,
                customer_remittance_address=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
                vat_number=None,
            )

            merchant_information = MerchantInformationInvoice(
                merchant_name=response_parsed["vendor_and_customer"]["sender_name"],
                merchant_address=response_parsed["vendor_and_customer"][
                    "sender_address"
                ],
                merchant_email=response_parsed["vendor_and_customer"]["sender_email"],
                merchant_tax_id=response_parsed["vendor_and_customer"]["sender_vat_id"],
                # Not supported by Rossum
                # -----------------------
                merchant_phone=None,
                merchant_fax=None,
                merchant_website=None,
                merchant_siren=None,
                merchant_siret=None,
                abn_number=None,
                gst_number=None,
                vat_number=None,
                pan_number=None,
                # -----------------------
            )

            locale_information = LocaleInvoice(
                language=response_parsed["basic_information"]["language"],
                currency=response_parsed["vat_and_amount"]["currency"],
            )

            taxes = []
            for tax in response_parsed["vat_and_amount"]["tax_details"]:
                taxes.append(
                    TaxesInvoice(
                        value=(
                            float(tax["tax_detail_tax"])
                            if tax["tax_detail_tax"]
                            else None
                        ),
                        rate=(
                            float(tax["tax_detail_rate"])
                            if tax["tax_detail_rate"]
                            else None
                        ),
                    )
                )

            bank_information = BankInvoice(
                account_number=response_parsed["payment_information"]["account_num"],
                iban=response_parsed["payment_information"]["iban"],
                sort_code=None,
                routing_number=None,
                swift=None,
                bsb=None,
                vat_number=None,
                rooting_number=None,
            )

            item_lines = []
            for item in response_parsed["line_items"]:
                item_lines.append(
                    ItemLinesInvoice(
                        unit_price=(
                            float(item["item_amount_base"])
                            if item["item_amount_base"]
                            else None
                        ),
                        amount=(
                            float(item["item_amount"]) if item["item_amount"] else None
                        ),
                        description=item["item_description"],
                        quantity=(
                            float(item["item_quantity"])
                            if item["item_quantity"]
                            else None
                        ),
                    )
                )

            extracted_data.append(
                InfosInvoiceParserDataClass(
                    customer_information=customer_information,
                    merchant_information=merchant_information,
                    locale=locale_information,
                    taxes=taxes,
                    bank_informations=bank_information,
                    item_lines=item_lines,
                    invoice_number=response_parsed["basic_information"][
                        "delivery_note_id"
                    ],
                    invoice_date=response_parsed["basic_information"]["date_issue"],
                    payment_terms=response_parsed["payment_information"]["terms"],
                    invoice_total=response_parsed["vat_and_amount"]["amount_total"],
                    amount_due=response_parsed["vat_and_amount"]["amount_due"],
                    purchase_order_number=response_parsed["basic_information"][
                        "order_id"
                    ],
                )
            )

        return InvoiceParserDataClass(extracted_data=extracted_data)

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        with open(file, "rb") as file_:
            _, annotation_endpoint = self._upload(file_)
            id, status = self._get_status_and_id(annotation_endpoint)
            while status != "to_review":
                sleep(1)
                id, status = self._get_status_and_id(annotation_endpoint)
                if status == "failed_import":
                    raise ProviderException(
                        "Invalid file, please check the file format."
                    )

        original_response = self._download_reviewing_data(id)
        standardized_response = self._invoice_standardization(original_response)

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
