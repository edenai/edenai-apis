from typing import Optional
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class CohereApi(ProviderInterface, TextInterface):
    provider_name = "cohere"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.base_url = self.api_settings["url"]
        self.headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {self.api_key}',
            'content-type': 'application/json',
        }

    def text__generation(
        self, text : str, 
        max_tokens : int,
        temperature :float,
        model : Optional[str] = None,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}generate"
        
        if not model:
            model = 'xlarge'
            
        payload = {
            "prompt": text,
            "model" : model,
            "temperature" : temperature,
            "max_tokens" : max_tokens,
        }
        original_response = requests.post(url, json=payload, headers= self.headers).json()
        
        if "message" in original_response:
            raise ProviderException(original_response["message"])
        
        standardized_response = GenerationDataClass(
            generated_text = original_response["text"]
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )