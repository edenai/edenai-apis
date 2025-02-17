from typing import Dict, Optional, Sequence

import requests

from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import (
    LogoDetectionDataClass,
    LogoBoundingPoly,
    LogoVertice,
    LogoItem,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import upload_file_to_s3


class SmartClickApi(ProviderInterface, ImageInterface):
    provider_name = "smartclick"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.key = self.api_settings["key"]
        self.base_url = "https://r-api.starla.ai/"
        self.headers = {
            "content-type": "application/json",
            "api-token": self.key,
        }

    def image__logo_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[LogoDetectionDataClass]:
        url = f"{self.base_url}logo-detection"

        # Get URL for the image
        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(file, file)

        payload = {"url": content_url}
        response = requests.request("POST", url, json=payload, headers=self.headers)

        if response.status_code != 200:
            # Poorly documented
            # ref: https://smartclick.ai/api/logo-detection/
            raise ProviderException(message=response.text, code=response.status_code)

        # standardized response : description/score/bounding_box
        items: Sequence[LogoItem] = []
        boxes = response.json()
        for box in boxes.get("bboxes"):
            vertices = []
            vertices.append(LogoVertice(x=box[0], y=box[1]))
            vertices.append(LogoVertice(x=box[2], y=box[1]))
            vertices.append(LogoVertice(x=box[2], y=box[3]))
            vertices.append(LogoVertice(x=box[0], y=box[3]))

            items.append(
                LogoItem(
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                    description=None,
                    score=None,
                )
            )
        standardized = LogoDetectionDataClass(items=items)
        return ResponseType[LogoDetectionDataClass](
            original_response=response.json(), standardized_response=standardized
        )
