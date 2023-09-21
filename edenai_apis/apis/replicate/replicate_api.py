from typing import Dict, List, Literal, Optional
import requests
from edenai_apis.features.image.generation.generation_dataclass import GenerationDataClass, GeneratedImageDataClass
from edenai_apis.features import ProviderInterface, TextInterface, ImageInterface
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass, GenerationDataClass as TextGenerationDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
import base64
from .config import get_model_id

class ReplicateApi(ProviderInterface, ImageInterface, TextInterface):
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

    def __get_response(
        self,
        url: str,
        payload: dict) -> dict:
        # Launch job 
        launch_job_response = requests.post(url, headers=self.headers, json = payload)
        try:
            launch_job_response_dict = launch_job_response.json()
        except requests.JSONDecodeError:
            raise ProviderException(launch_job_response.text, code=launch_job_response.status_code)
        if launch_job_response.status_code != 201:
            raise ProviderException(launch_job_response_dict.get("detail"), code=launch_job_response.status_code)
        
        url_get_response = launch_job_response_dict["urls"]["get"]
        
        # Get job response
        response = requests.get(url_get_response, headers=self.headers)
        response_dict = response.json()
        if response.status_code != 200:
            raise ProviderException(response_dict.get("detail"), code=response.status_code)
        
        status = response_dict["status"]
        while status != "succeeded": 
            response = requests.get(url_get_response, headers=self.headers)

            try:
                response_dict = response.json()
            except requests.JSONDecodeError:
                raise ProviderException(response.text, code=response.status_code)

            if response.status_code != 200:
                raise ProviderException(response_dict.get("error", response_dict), code=response.status_code)

            status = response_dict["status"]
            
        return response_dict
            
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
        
        response_dict= ReplicateApi.__get_response(self, url, payload)
        image_url = response_dict.get("output")
        image_bytes = base64.b64encode(requests.get(image_url).content)
        
        return ResponseType[GenerationDataClass](
            original_response=response_dict,
            standardized_response=GenerationDataClass(
                items=[
                    GeneratedImageDataClass(
                        image=image_bytes, image_resource_url=image_url
                    )
                ]
            )
        )
    
    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[ChatDataClass]:
        # Construct the API URL
        url = f"{self.base_url}/predictions"
        
        # Get the model ID based on the provided model name
        model_id = get_model_id[model]
        
        # Build the prompt by formatting the previous history and current text
        prompt = ''
        if previous_history:
            for msg in previous_history:
                if msg["role"] == "user":
                    prompt += "\n[INST]" + msg["message"] + "[/INST]\n"
                else:
                    prompt += "\n" + msg["message"] + "\n"
        
        prompt += "\n[INST]" + text + "[/INST]\n"
        
        # Construct the payload for chat interaction
        payload = {
            "input": {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "min_new_tokens": -1
            },
            "version": model_id
        }
        
        # Include system prompt if provided
        if chatbot_global_action: 
            payload["input"]["system_prompt"] = chatbot_global_action
        
        # Call the API and get the response dictionary
        get_response_dict = ReplicateApi.__get_response(self, url, payload)
        
        # Extract generated text from the API response
        generated_text = ''.join(get_response_dict.get('output', ['']))
        
        # Build a list of ChatMessageDataClass objects for the conversation history
        message = [
            ChatMessageDataClass(role="user", message=text),
            ChatMessageDataClass(role="assistant", message=generated_text),
        ]
        
        # Build the standardized response
        standardized_response = ChatDataClass(
            generated_text=generated_text,
            message=message
        )
        
        return ResponseType[ChatDataClass](
            original_response=get_response_dict,
            standardized_response=standardized_response,
        )
