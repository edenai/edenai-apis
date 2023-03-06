import requests
import base64
from typing import Literal
from edenai_apis.utils.types import ResponseType
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass
)

class DeepAIApi(ProviderInterface, ImageInterface):
    provider_name = "deepai"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Api-Key": f"{self.api_key}",
        }
        
    def image__generation(
        self,
        text: str,
        resolution : Literal["256x256", "512x512","1024x1024"],
        num_images: int = 1) -> ResponseType[GenerationDataClass]:
        url = 'https://api.deepai.org/api/text2img'
        try:
            size = resolution.split("x")
            payload = {
                'text' : text,
                'grid_size': '1',
                'width' : int(size[0]),
                'height' : int(size[1]),
            }
            original_response = requests.post(
                url, data=payload, headers=self.headers
            ).json()
        except Exception as exc:
            raise ProviderException(str(exc)) from exc
        
        if 'err' in original_response:
            raise ProviderException(original_response['err'])
        
        image_url = original_response.get('output_url')
        image_bytes = base64.b64encode(requests.get(image_url).content)
        
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(items = [
                GeneratedImageDataClass(
                    image = image_bytes,
                    image_resource_url = image_url
                    )
                    ]
                )
            )