import requests
from typing import Sequence, Literal
from edenai_apis.utils.types import ResponseType
from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.features.image.generation import (
    GenerationDataClass,
    GeneratedImageDataClass
)

class StabilityAIApi(ProviderInterface, ImageInterface):
    provider_name = "stabilityai"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512","1024x1024"],
        num_images: int = 1
        ) -> ResponseType[GenerationDataClass]:
        url = "https://api.stability.ai/v1beta/generation/stable-diffusion-512-v2-0/text-to-image"
        size = resolution.split("x")
        payload = {
        "text_prompts": [
            {
                "text": text,
            }
        ],
        "width": int(size[0]),
        "height" : int(size[1]),
        "samples": num_images,
        }
        
        original_response = requests.post(url, headers=self.headers, json=payload).json()
        
        # Handle error
        if "message" in original_response:
            raise ProviderException(original_response['message'])
        
        generations: Sequence[GeneratedImageDataClass] = []
        for generated_image in original_response.get('artifacts'):
            generations.append(GeneratedImageDataClass(
                image=generated_image.get('base64')))
            
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(items = generations))
        
    