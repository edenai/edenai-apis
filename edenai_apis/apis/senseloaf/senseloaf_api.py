import mimetypes
from typing import Dict

import aiofiles

from edenai_apis.features import OcrInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.ocr import ResumeParserDataClass
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.file_handling import FileHandler

from .client import Client, Parser
from .client_async import AsyncClient
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

    async def ocr__aresume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:
        file_handler = FileHandler()
        file_wrapper = None

        try:
            if not file:
                if not file_url:
                    raise ProviderException(
                        "Either file or file_url must be provided", code=400
                    )
                file_wrapper = await file_handler.download_file(file_url)
                file_content = await file_wrapper.get_bytes()
                mime_type = file_wrapper.file_info.file_media_type
                extension = file_wrapper.file_info.file_extension
                file = f"resume.{extension}"
            else:
                async with aiofiles.open(file, "rb") as file_:
                    file_content = await file_.read()
                mime_type = mimetypes.guess_type(file)[0]

            async with await AsyncClient.create(
                email=self.api_settings.get("email"),
                password=self.api_settings.get("password"),
            ) as async_client:
                original_response = await async_client.parse_document_async(
                    parse_type=Parser.RESUME,
                    file=file,
                    file_content=file_content,
                    mime_type=mime_type,
                )

            mapper = ResumeMapper(original_response)

            return ResponseType[ResumeParserDataClass](
                original_response=mapper.original_response(),
                standardized_response=mapper.standard_response(),
            )
        finally:
            if file_wrapper:
                file_wrapper.close_file()
