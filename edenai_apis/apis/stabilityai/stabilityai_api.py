import base64
from io import BytesIO
import json
import requests
from typing import Dict, Sequence, Literal
from edenai_apis.utils.types import ResponseType
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class StabilityAIApi(ProviderInterface, ImageInterface):
    provider_name = "stabilityai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
    ) -> ResponseType[GenerationDataClass]:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-512-v2-1/text-to-image"
        size = resolution.split("x")
        payload = {
            "text_prompts": [
                {
                    "text": text,
                }
            ],
            "width": int(size[0]),
            "height": int(size[1]),
            "samples": num_images,
        }

        try:
            response = requests.post(
                url, headers=self.headers, json=payload
            )
            original_response = response.json()
        except json.JSONDecodeError:
            raise ProviderException("Internal Server Error", code=500)

        # Handle error
        if "message" in original_response:
            raise ProviderException(
                original_response["message"],
                code = response.status_code
            )

        generations: Sequence[GeneratedImageDataClass] = []
        for generated_image in original_response.get("artifacts"):
            image_b64 = generated_image.get("base64")

            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                GeneratedImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(items=generations),
        )
