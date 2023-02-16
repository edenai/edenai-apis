from io import BufferedReader
from json import JSONDecodeError

import requests
from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import InvoiceParserDataClass
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

    def ocr__invoice_parser(self, file: BufferedReader, language: str) -> ResponseType[InvoiceParserDataClass]:
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
            raise ProviderException(message="Internal Server Error", code=500)

        if response != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

