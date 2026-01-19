import base64
import json
from typing import Dict, Optional, Any
import aiofiles
import requests
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import BackgroundRemovalDataClass
from edenai_apis.features.image.utils.upload import aget_resource_url
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.file_handling import FileHandler
from edenai_apis.utils.http_client import async_client, IMAGE_TIMEOUT
from edenai_apis.utils.types import ResponseType


class ClipdropApi(ProviderInterface, ImageInterface):
    provider_name = "clipdrop"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "x-api-key": self.api_settings["api_key"],
            "Accept": "image/png",
        }

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        url = "https://clipdrop-api.co/remove-background/v1"
        with open(file, "rb") as f:
            files = {"image_file": f.read()}

            response = requests.post(url, files=files, headers=self.headers)

        if response.status_code != 200:
            try:
                error_message = response.json()["error"]
            except (KeyError, json.JSONDecodeError):
                error_message = "Internal Server Error"
            raise ProviderException(error_message, code=response.status_code)

        image_b64 = base64.b64encode(response.content).decode("utf-8")
        resource_url = BackgroundRemovalDataClass.generate_resource_url(image_b64)

        return ResponseType[BackgroundRemovalDataClass](
            original_response=response.text,
            standardized_response=BackgroundRemovalDataClass(
                image_b64=image_b64,
                image_resource_url=resource_url,
            ),
        )

    async def image__abackground_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        url = "https://clipdrop-api.co/remove-background/v1"
        file_handler = FileHandler()
        file_wrapper = None  # Track for cleanup

        try:
            if not file:
                # try to use the url
                if not file_url:
                    raise ProviderException(
                        "Either file or file_url must be provided", code=400
                    )
                file_wrapper = await file_handler.download_file(file_url)
                image_file = await file_wrapper.get_bytes()
            else:
                async with aiofiles.open(file, "rb") as f:
                    image_file = await f.read()
            async with async_client(IMAGE_TIMEOUT) as client:
                response = await client.post(
                    url, files={"image_file": image_file}, headers=self.headers
                )
                if response.status_code != 200:
                    try:
                        error_message = response.json()["error"]
                    except (KeyError, json.JSONDecodeError):
                        error_message = "Internal Server Error"
                    raise ProviderException(error_message, code=response.status_code)

                image_b64 = base64.b64encode(response.content).decode("utf-8")

                resource_url_dict = await aget_resource_url(image_b64)

                return ResponseType[BackgroundRemovalDataClass](
                    original_response=response.text,
                    standardized_response=BackgroundRemovalDataClass(
                        **resource_url_dict
                    ),
                )
        finally:
            # Clean up temp file if it was created
            if file_wrapper:
                file_wrapper.close_file()
