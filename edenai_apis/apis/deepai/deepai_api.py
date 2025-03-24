import base64
import json
from typing import Dict, Literal, Optional

import requests

from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.generation import (
    GeneratedImageDataClass,
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.llmengine.utils.moderation import moderate


class DeepAIApi(ProviderInterface, ImageInterface):
    provider_name = "deepai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Api-Key": f"{self.api_key}",
        }

    @moderate
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        url = "https://api.deepai.org/api/text2img"
        size = resolution.split("x")
        payload = {
            "text": text,
            "grid_size": "1",
            "width": int(size[0]),
            "height": int(size[1]),
        }
        response = requests.post(url, data=payload, headers=self.headers)
        try:
            original_response = response.json()
        except requests.JSONDecodeError:
            raise ProviderException(response.text, code=response.status_code)

        if not response.ok:
            err_msg = original_response.get("err") or json.dumps(original_response)
            raise ProviderException(err_msg, response.status_code)

        image_url = original_response.get("output_url")
        image_response = requests.get(image_url)
        if not image_response.ok:
            raise ProviderException(
                image_response.text, code=image_response.status_code
            )
        image_bytes = base64.b64encode(image_response.content)

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(
                items=[
                    GeneratedImageDataClass(
                        image=image_bytes, image_resource_url=image_url
                    )
                ]
            ),
        )
