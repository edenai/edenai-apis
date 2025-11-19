from http import HTTPStatus
import json
from typing import Optional, Dict, Any, Union

import aiofiles
import httpx
import requests

from edenai_apis.apis.photoroom.types import PhotoroomBackgroundRemovalParams
from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.background_removal import BackgroundRemovalDataClass
from edenai_apis.features.image.utils.upload import aget_resource_url
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.file_handling import FileHandler
from edenai_apis.utils.types import ResponseType


class PhotoroomApi(ImageInterface, ProviderInterface):
    provider_name: str = "photoroom"

    def __init__(self, api_key: Optional[Dict[str, str]] = None) -> None:
        self.api_settings: Dict[str, str] = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_key or {}
        )

        self.api_key: str = self.api_settings["api_key"]
        self.headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "accept": "application/json",
        }
        self.base_url: str = "https://sdk.photoroom.com/v1/"

    @staticmethod
    def _handle_error(response: Union[requests.Response, httpx.Response]) -> None:
        if response.status_code == HTTPStatus.OK:
            pass
        elif response.status_code in [
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.PAYMENT_REQUIRED,
        ]:
            raise ProviderException(
                message=response.json()["detail"], code=response.status_code
            )
        else:
            raise ProviderException(
                message="Provider has not found a background of the image.",
                code=response.status_code,
            )

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        with open(file, "rb") as f:
            files = {"image_file": f.read()}

            if provider_params is None or not isinstance(provider_params, dict):
                photoroom_params = PhotoroomBackgroundRemovalParams()
            else:
                photoroom_params = PhotoroomBackgroundRemovalParams(**provider_params)

            response = requests.post(
                f"{self.base_url}segment",
                headers=self.headers,
                files=files,
                data=photoroom_params.model_dump(),
            )

            PhotoroomApi._handle_error(response)
            original_response = response.json()

            img_b64 = original_response["result_b64"]
            resource_url = BackgroundRemovalDataClass.generate_resource_url(
                img_b64,
                fmt=photoroom_params.format,
            )

            return ResponseType[BackgroundRemovalDataClass](
                original_response=original_response,
                standardized_response=BackgroundRemovalDataClass(
                    image_b64=img_b64,
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
        file_handler = FileHandler()
        file_wrapper = None  # Track for cleanup

        try:
            if provider_params is None or not isinstance(provider_params, dict):
                photoroom_params = PhotoroomBackgroundRemovalParams()
            else:
                photoroom_params = PhotoroomBackgroundRemovalParams(**provider_params)

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

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, read=120.0)
            ) as client:
                response = await client.post(
                    f"{self.base_url}segment",
                    files={"image_file": image_file},
                    headers=self.headers,
                    data=photoroom_params.model_dump(),
                )
                PhotoroomApi._handle_error(response)
                original_response = response.json()

                image_b64 = original_response["result_b64"]

                resource_url_dict = await aget_resource_url(
                    image_b64, photoroom_params.format
                )

                return ResponseType[BackgroundRemovalDataClass](
                    original_response=original_response,
                    standardized_response=BackgroundRemovalDataClass(
                        **resource_url_dict
                    ),
                )
        finally:
            # Clean up temp file if it was created
            if file_wrapper:
                file_wrapper.close_file()
