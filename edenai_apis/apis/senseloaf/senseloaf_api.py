from typing import Dict

from edenai_apis.features import OcrInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.ocr import ResumeParserDataClass

from .client import Client, Parser
from .remapping import ResumeMapper


class SenseloafApi(ProviderInterface, OcrInterface):
    provider_name = "senseloaf"

    def __init__(self, api_keys: Dict = {}):
        super().__init__()
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.client = Client(
            self.api_settings.get("api_key", None),
            self.api_settings.get("email", None),
            self.api_settings.get("password", None),
        )

    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:

        original_response = self.client.parse_document(
            parse_type=Parser.RESUME, file=file, url=file_url
        )

        mapper = ResumeMapper(original_response)

        return ResponseType[ResumeParserDataClass](
            original_response=mapper.original_response(),
            standardized_response=mapper.standard_response(),
        )
