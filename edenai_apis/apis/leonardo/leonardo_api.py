import base64
import http.client
from typing import Dict, Generator, List, Literal, Optional, Union, overload
import requests

from edenai_apis.features import ImageInterface
from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from .config import get_model_id_image
from edenai_apis.utils.parsing import extract
from edenai_apis.llmengine.utils.moderation import moderate


class LeonardoApi(ProviderInterface, ImageInterface):
    provider_name = "leonardo"

    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Token {api_settings['api_key']}",
        }
        self.base_url = "https://cloud.leonardo.ai/api/rest/v1"

    @overload
    def __get_response(self, url: str, payload: dict) -> Generator: ...

    @overload
    def __get_response(self, url: str, payload: dict) -> dict: ...

    def __get_response(self, url: str, payload: dict) -> Union[Generator, dict]:
        # Launch job

        try:
            launch_job_response = requests.post(url, headers=self.headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ProviderException(e)

        try:
            launch_job_response_dict = launch_job_response.json()
        except requests.JSONDecodeError:
            raise ProviderException(
                launch_job_response.text, code=launch_job_response.status_code
            )
        if launch_job_response.status_code != 200:
            raise ProviderException(
                launch_job_response_dict.get("error", launch_job_response_dict),
                code=launch_job_response.status_code,
            )

        generation_id = launch_job_response_dict["sdGenerationJob"]["generationId"]
        url_get_response = f"{self.base_url}/generations/{generation_id}"

        # Get job response
        response = requests.get(url_get_response, headers=self.headers)

        if response.status_code >= 500:
            raise ProviderException(
                message=http.client.responses[response.status_code],
                code=response.status_code,
            )
        try:
            response_dict = response.json()
        except requests.JSONDecodeError:
            raise ProviderException(f"Invalid JSON response: {response.text}")

        if response.status_code != 200:
            raise ProviderException(
                response_dict.get("detail"), code=response.status_code
            )

        status = response_dict["generations_by_pk"]["status"]
        while status != "COMPLETE":
            response = requests.get(url_get_response, headers=self.headers)
            try:
                response_dict = response.json()
            except requests.JSONDecodeError:
                raise ProviderException(response.text, code=response.status_code)

            if response.status_code != 200:
                raise ProviderException(
                    response_dict.get("error", response_dict), code=response.status_code
                )

            status = response_dict["generations_by_pk"]["status"]

        return response_dict

    @moderate
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        size = resolution.split("x")
        payload = {
            "prompt": text,
            "width": int(size[0]),
            "height": int(size[1]),
            "modelId": get_model_id_image.get(model, model),
            "num_images": num_images,
            "ultra": False,  # True == High quality, False == Low quality
            "alchemy": False,  # True == Quality, False == Speed
            "contrast": 3.5,  # low contrast : 3, medium contrast : 3.5, high contrast : 4
            "styleUUID": None,
        }

        url = f"{self.base_url}/generations"

        response_dict = LeonardoApi.__get_response(self, url, payload)
        generation_by_pk = response_dict.get("generations_by_pk", {}) or {}
        generated_images = generation_by_pk.get("generated_images", []) or []
        image_url = [image.get("url") for image in generated_images]

        generated_images = []
        if isinstance(image_url, list):
            for image in image_url:
                generated_images.append(
                    GeneratedImageDataClass(
                        image=base64.b64encode(requests.get(image).content),
                        image_resource_url=image,
                    )
                )
        else:
            generated_images.append(
                GeneratedImageDataClass(
                    image=base64.b64encode(requests.get(image_url).content),
                    image_resource_url=image_url,
                )
            )

        return ResponseType[GenerationDataClass](
            original_response=response_dict,
            standardized_response=GenerationDataClass(items=generated_images),
        )
