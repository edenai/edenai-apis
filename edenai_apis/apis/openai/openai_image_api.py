import base64
from io import BytesIO
from typing import Sequence, Literal, Optional

import requests

from edenai_apis.features import ImageInterface
from edenai_apis.features.image.generation import (
    GenerationDataClass as ImageGenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .helpers import (
    get_openapi_response,
)


class OpenaiImageApi(ImageInterface):
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None
    ) -> ResponseType[ImageGenerationDataClass]:
        url = f"{self.url}/images/generations"
        payload = {
            "prompt": text,
            "model": model,
            "n": num_images,
            "size": resolution,
            "response_format": "b64_json",
        }
        response = requests.post(
            url, json=payload, headers=self.headers
        )
        original_response = get_openapi_response(response)

        generations: Sequence[GeneratedImageDataClass] = []
        for generated_image in original_response.get("data"):
            image_b64 = generated_image.get("b64_json")

            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                GeneratedImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[ImageGenerationDataClass](
            original_response=original_response,
            standardized_response=ImageGenerationDataClass(items=generations),
        )
