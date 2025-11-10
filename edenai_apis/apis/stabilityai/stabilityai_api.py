import asyncio
import base64
import json
from io import BytesIO
from json import JSONDecodeError
from typing import Dict, Literal, Optional, Any, List, Sequence

import httpx
import requests

from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import BackgroundRemovalDataClass
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.image.variation import (
    VariationDataClass,
    VariationImageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    aupload_file_bytes_to_s3,
    upload_file_bytes_to_s3,
)
from edenai_apis.llmengine.utils.moderation import async_moderate, moderate


class StabilityAIApi(ProviderInterface, ImageInterface):
    provider_name = "stabilityai"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
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
        url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
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
            response = requests.post(url, headers=self.headers, json=payload)
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        # Handle error
        if response.status_code != 200:
            raise ProviderException(
                message=original_response.get("message"), code=response.status_code
            )

        generations: List[GeneratedImageDataClass] = []
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

    @async_moderate
    async def image__ageneration(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
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

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120)) as client:
            response = await client.post(url, json=payload, headers=self.headers)

            try:
                original_response = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException("Internal Server Error", code=500) from exc

            # Handle error
            if response.is_error:
                raise ProviderException(
                    message=original_response.get("message"), code=response.status_code
                )

            async def process_and_upload_image(image_data: dict):
                image = image_data.get("base64")

                # Decode base64 in thread pool (CPU-bound operation)
                def decode_image():
                    base64_bytes = image.encode()
                    return BytesIO(base64.b64decode(base64_bytes))

                image_bytes = await asyncio.to_thread(decode_image)

                # Upload to S3 asynchronously
                resource_url = await aupload_file_bytes_to_s3(
                    image_bytes, ".png", USER_PROCESS
                )

                return GeneratedImageDataClass(
                    image=image, image_resource_url=resource_url
                )

            # Process and upload all images concurrently
            generated_images = await asyncio.gather(
                *[
                    process_and_upload_image(image)
                    for image in original_response.get("artifacts", [])
                ],
                return_exceptions=True
            )

            # Filter out exceptions and log/handle them
            successful_images = []
            for img in generated_images:
                if isinstance(img, Exception):
                    # Log or handle the exception
                    raise ProviderException(f"Failed to process image: {str(img)}")
                successful_images.append(img)

            return ResponseType[GenerationDataClass](
                original_response=original_response,
                standardized_response=GenerationDataClass(items=successful_images),
            )

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        url = "https://api.stability.ai/v2beta/stable-image/edit/remove-background"
        with open(file, "rb") as f:
            files = {"image": f.read()}
            headers = {"Authorization": f"Bearer {self.api_key}", "accept": "image/*"}

            response = requests.post(url, files=files, headers=headers)
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        image_b64 = base64.b64encode(response.content).decode("utf-8")
        resource_url = BackgroundRemovalDataClass.generate_resource_url(image_b64)

        return ResponseType[BackgroundRemovalDataClass](
            original_response=response.text,
            standardized_response=BackgroundRemovalDataClass(
                image_b64=image_b64,
                image_resource_url=resource_url,
            ),
        )

    def image__variation(
        self,
        file: str,
        prompt: Optional[str],
        num_images: Optional[int] = 1,
        resolution: Literal["256x256", "512x512", "1024x1024"] = "512x512",
        temperature: Optional[float] = 0.3,
        model: Optional[str] = None,
        file_url: str = "",
        **kwargs,
    ) -> ResponseType[VariationDataClass]:
        url = f"https://api.stability.ai/v1/generation/{model}/image-to-image"
        del self.headers["Content-Type"]
        prompt = prompt or ""
        with open(file, "rb") as img:

            if not prompt:
                prompt = "Generate a variation of this image and maintain the style"
            data = {
                "image_strength": 1 - temperature,
                "text_prompts[0][text]": prompt,
                "samples": num_images,
            }
            files = {"init_image": img}

            response = requests.post(url, headers=self.headers, data=data, files=files)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)

        else:
            try:
                original_response = response.json()
            except JSONDecodeError:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )

            generations: Sequence[VariationImageDataClass] = []

            for generated_image in original_response.get("artifacts"):
                image_b64 = generated_image.get("base64")
                image_data = image_b64.encode()
                image_content = BytesIO(base64.b64decode(image_data))

                resource_url = upload_file_bytes_to_s3(
                    image_content, ".png", USER_PROCESS
                )
                generations.append(
                    VariationImageDataClass(
                        image=image_b64, image_resource_url=resource_url
                    )
                )
            return ResponseType[VariationDataClass](
                original_response=original_response,
                standardized_response=VariationDataClass(items=generations),
            )
