from typing import Optional, List
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    GenerationDataClass,
    ItemCustomClassificationDataClass,
    CustomClassificationDataClass,
    SummarizeDataClass,
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
            'Cohere-Version': '2022-12-06',
        }

    def _calculate_summarize_length(output_sentences : int):
        if output_sentences < 3:
            return 'short'
        elif output_sentences < 6:
            return 'medium'
        elif output_sentences > 6:
            return 'long'
        
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
        }
        
        if max_tokens !=0:
            payload['max_tokens'] = max_tokens
            
        original_response = requests.post(url, json=payload, headers= self.headers).json()
        
        if "message" in original_response:
            raise ProviderException(original_response["message"])
        
        generated_texts = original_response.get('generations')
        standardized_response = GenerationDataClass(
            generated_text = generated_texts[0]['text']
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )
        
    def text__custom_classification(
        self,
        texts: List[str],
        labels: List[str],
        examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        
        # Build the request
        url = f"{self.base_url}classify"
        example_dict = []
        for example in examples:
            example_dict.append(
                {
                    'text' : example[0],
                    'label' : example[1]
                }
            )
        payload = {
            "inputs": texts,
            "examples" : example_dict,
            "model" : 'large',
        }
        original_response = requests.post(url, json=payload, headers= self.headers).json()
        
        # Handle provider errors
        if "message" in original_response:
            raise ProviderException(original_response["message"])
        
        # Standardization 
        classifications = []
        for classification in original_response.get('classifications'):
            classifications.append(
                ItemCustomClassificationDataClass(
                    input = classification['input'],
                    label = classification['prediction'],
                    confidence = classification['confidence'],
                )
            )

        return ResponseType[CustomClassificationDataClass](
            original_response=original_response,
            standardized_response = CustomClassificationDataClass(classifications = classifications)
        )
    
    def text__summarize(self, text: str, output_sentences: int, language: str, model: Optional[str]) -> ResponseType[SummarizeDataClass]:
        url = f"{self.base_url}summarize"
        length = 'long'
        if not model:
            model = 'summarize-xlarge'
        if output_sentences:
            length = CohereApi._calculate_summarize_length(output_sentences)
            
        payload = {
                "length": length,
                "format": "paragraph",
                "model": model,
                "extractiveness": "low",
                "temperature": 0.3,
                "text": text,
            }

        original_response = requests.post(url, json=payload, headers= self.headers).json()
        
        if "message" in original_response:
            raise ProviderException(original_response["message"])
        
        standardized_response = SummarizeDataClass(result=original_response.get("summary", {}))
        
        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )
        