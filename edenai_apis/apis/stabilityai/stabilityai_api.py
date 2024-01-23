import base64
import json
from io import BytesIO
from typing import Dict, Literal, Optional, Any, List, Sequence

import requests

from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import BackgroundRemovalDataClass
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3

from edenai_apis.features.image.variation import (
    VariationDataClass, VariationImageDataClass
)

from stability_sdk import client
from stability_sdk.api import (
    ClassifierException, OutOfCreditsException
)
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from edenai_apis.features.image.variation import (
    VariationDataClass, VariationImageDataClass
)
from PIL import Image

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

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
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
        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
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

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        url = "https://clipdrop-api.co/remove-background/v1"
        with open(file, "rb") as f:
            files = {"image_file": f.read()}
            headers = {
                "x-api-key": self.api_settings["bg_removal_api_key"],
                "Accept": "image/png",
            }

            response = requests.post(url, files=files, headers=headers)

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

    def image__variation (self, 
        file :str, 
        prompt : Optional[str] = "",  
        num_images : Optional[int] = 1, 
        resolution : Literal["256x256", "512x512", "1024x1024"] = "512x512", 
        temperature : Optional[float] = 0.3,
        ) ->ResponseType[VariationImageDataClass]:

        stabilityapi = client.StabilityInference(
            key = self.api_key,
            verbose=True,
            engine = 'stable-diffusion-xl-1024-v1-0'    
        )

        size = resolution.split('x')
        sizew = size[0]
        sizeh = size[1]

        with open(file, 'rb') as fstream :
            image = Image.open(BytesIO(fstream.read()))

        if prompt == "" :
            prompt = "Generate a variation of this image and maintain the style"

        try :
            response = stabilityapi.generate (
                prompt = prompt,
                start_schedule=temperature, 
                init_image=image,
                width=int(sizew),    
                height=int(sizeh),
                samples=num_images,
            )
        except OutOfCreditsException as crediterror:
            raise ProviderException(message = crediterror.details)
        except ClassifierException as message_error :
            raise ProviderException(message = message_error.classifier_result)

        for resp in response:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    raise ProviderException(
                        message=artifact.finish_reason)
                if artifact.type == generation.ARTIFACT_IMAGE:
                    original_response = response
                    generations: Sequence[VariationImageDataClass] = []

                    image_content = BytesIO(base64.b64decode(artifact.binary))
                    resource_url = upload_file_bytes_to_s3(artifact.binary, ".png", USER_PROCESS)
                    generations.append(
                        VariationImageDataClass(
                            image=str(image_content), image_resource_url=resource_url
                        )
                    )
                    return ResponseType[VariationDataClass](
                        original_response=original_response,
                        standardized_response=VariationDataClass(items=generations),
                    )