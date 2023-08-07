from typing import Dict, Literal
import requests
from edenai_apis.features.image.generation.generation_dataclass import GenerationDataClass, GeneratedImageDataClass
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
import base64

class ReplicateApi(ProviderInterface, ImageInterface):
    provider_name = "replicate"

    def __init__(self, api_keys: Dict = {}):
        api_settings = load_provider(
            ProviderDataEnum.KEY, provider_name=self.provider_name, api_keys=api_keys
        )
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization" : f"Token {api_settings['api_key']}"
        }
        self.base_url = "https://api.replicate.com/v1"

    def image__generation(
        self, 
        text: str, 
        resolution: Literal['256x256', '512x512', '1024x1024'], 
        num_images: int = 1) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}/predictions"
        size = resolution.split("x")
        payload = {
            "input" : {
                "prompt" : text,
                "width" : int(size[0]),
                "height" : int(size[1]),
            },
            "version": "c0259010b93e7a4102a4ba946d70e06d7d0c7dc007201af443cfc8f943ab1d3c"
        }
        
        # Launch job 
        launch_job_response = requests.post(url, headers=self.headers, json = payload)
        launch_job_response_dict = launch_job_response.json()
        if launch_job_response.status_code != 201:
            raise ProviderException(launch_job_response_dict.get("detail"), code=launch_job_response.status_code)
        
        url_get_response = launch_job_response_dict["urls"]["get"]
        
        # Get job response
        get_response = requests.get(url_get_response, headers=self.headers)
        get_response_dict = get_response.json()
        if get_response.status_code != 200:
            raise ProviderException(get_response_dict.get("detail"), code=get_response.status_code)
        
        status = get_response_dict["status"]
        while status != "succeeded": 
            get_response = requests.get(url_get_response, headers=self.headers)
            get_response_dict = get_response.json()
            if get_response.status_code != 200:
                raise ProviderException(get_response_dict["error"], code=get_response.status_code)
            status = get_response_dict["status"]

        image_url = get_response_dict.get("output")
        image_bytes = base64.b64encode(requests.get(image_url).content)
        
        return ResponseType[GenerationDataClass](
            original_response=get_response_dict,
            standardized_response=GenerationDataClass(
                items=[
                    GeneratedImageDataClass(
                        image=image_bytes, image_resource_url=image_url
                    )
                ]
            )
        )
