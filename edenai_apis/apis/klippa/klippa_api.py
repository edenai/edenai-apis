from io import BufferedReader
from json import JSONDecodeError

import requests
from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import BankInvoice, CustomerInformationInvoice, InfosInvoiceParserDataClass, ItemLinesInvoice, LocaleInvoice, MerchantInformationInvoice, TaxesInvoice
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException

class KlippaApi(ProviderInterface, OcrInterface):
    provider_name = "klippa"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["subscription_key"]
        self.url = self.api_settings["url"]

        self.headers = {
            "X-Auth-Key": self.api_key,
        }

    def _make_post_request(self, file: BufferedReader):
        files = {
            "document": file,
            "pdf_text_extraction": "full",
        }

        response = requests.post(
            url=self.url,
            headers=self.headers,
            files=files
        )

        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(message="Internal Server Error", code=500) from JSONDecodeError

        if response.status_code != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

    def ocr__invoice_parser(
        self,
        file: BufferedReader,
        language: str
    ) -> ResponseType[InvoiceParserDataClass]:
        original_response = self._make_post_request(file)

        data_response = original_response["data"]
        customer_information = CustomerInformationInvoice(
            customer_name=data_response["customer_name"],
            customer_address=data_response["customer_address"],
            customer_email=data_response["customer_email"],
            customer_phone=data_response["customer_phone"],
            customer_tax_id=data_response["customer_vat_number"],
            customer_id=data_response["customer_id"],
            customer_billing_address=data_response["customer_address"],
        )

        merchant_information = MerchantInformationInvoice(
            merchant_name=data_response["merchant_name"],
            merchant_address=data_response["merchant_address"],
            merchant_email=data_response["merchant_email"],
            merchant_phone=data_response["merchant_phone"],
            merchant_tax_id=data_response["merchant_vat_number"],
            merchant_id=data_response["merchant_id"],
            merchant_siret=data_response["merchant_coc_number"],
            merchant_website=data_response["merchant_website"],
        )

        bank_information = BankInvoice(
            account_number=data_response["merchant_bank_account_number"],
            sort_code=data_response["merchant_bank_domestic_bank_code"],
            iban=data_response["merchant_bank_account_number_bic"],
        )

        tax_information = TaxesInvoice(
            tax_rate=data_response["personal_income_tax_rate"],
            tax_amount=data_response["personal_income_tax_amount"],
        )

        locale_information = LocaleInvoice(
            currency=data_response["currency"],
            language=data_response["document_language"],
        )

        item_lines = []
        for item in data_response["lines"][0]['lineitems']:
            item_lines.append(ItemLinesInvoice(
                    description=item["description"],
                    quantity=item["quantity"],
                    unit_price=item["amount_each"],
                    discount=item["discount_amount"],
                    amount=item["amount"],
                    tax_rate=item["vat_percentage"],
                    tax_amount=item["vat_amount"],
                    product_code=item["sku"],
                )
            )

        standardize_response =InvoiceParserDataClass(extracted_data=[InfosInvoiceParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            bank_informations=bank_information,
            taxes=[tax_information],
            locale=locale_information,
            item_lines=item_lines,
            invoice_number=data_response["invoice_number"],
            invoice_date=data_response["date"],
            invoice_total=data_response["amount"],
        )])

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response
        )

    def ocr__receipt_parser(
        self,
        file: BufferedReader,
        language: str
    ) -> ResponseType[ReceiptParserDataClass]:
        return self._make_post_request(file)
