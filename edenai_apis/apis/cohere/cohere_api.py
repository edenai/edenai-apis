from typing import Optional, List, Dict
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    GenerationDataClass,
    ItemCustomClassificationDataClass,
    CustomClassificationDataClass,
    SummarizeDataClass,
    CustomNamedEntityRecognitionDataClass
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
import json


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
        
    def _format_custom_ner_examples(
        example : Dict
        ):
        # Get the text 
        text = example['text']
        
        # Get the entities 
        entities = example['entities']
        
        # Create an empty list to store the extracted entities
        extracted_entities = []
        
        # Loop through the entities and extract the relevant information
        for entity in entities:
            category = entity['category']
            entity_name = entity['entity']
            
            # Append the extracted entity to the list
            extracted_entities.append({'entity': entity_name, 'category': category})
            
        # Create the string with the extracted entities
        return f"""
        Text: #{text}#
        Answer: "[{', '.join([f'{{"entity":"{entity["entity"]}", "category":"{entity["category"]}"}}' for entity in extracted_entities])}]"
        ---
            """

    def text__generation(
        self, text : str, 
        max_tokens : int,
        temperature :float,
        model : str,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}generate"
        
          
        payload = {
            "prompt": text,
            "model" : model,
            "temperature" : temperature,
            "stop_sequences": ["--"],
            "frequency_penalty" : 0.3,
            "truncate": "END",
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
            "model" : 'base',
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
    
    def text__summarize(self, text: str, output_sentences: int, language: str, model: str) -> ResponseType[SummarizeDataClass]:
        url = f"{self.base_url}summarize"
        length = 'long'
        
        if output_sentences:
            length = CohereApi._calculate_summarize_length(output_sentences)
            
        payload = {
                "length": length,
                "format": "paragraph",
                "model": model,
                "extractiveness": "low",
                "temperature": 0.0,
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
        
    def text__custom_named_entity_recognition(
        self, 
        text: str, 
        entities: List[str],
        examples: Optional[List[Dict]] = None) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.base_url}generate"
        
        # Construct the prompt
        built_entities = ','.join(entities)
        prompt_examples = ''
        if examples is not None:
            for example in examples :       
                prompt_examples = prompt_examples + CohereApi._format_custom_ner_examples(example)
        prompt =f"""You act as a named entities recognition model. Extract the specified entities ({built_entities}) from the text enclosed in hash symbols (#) and return a JSON List of dictionaries with two keys: "entity" and "category". The "entity" key represents the detected entity and the "category" key represents the category of the entity.

If no entities are found, return an empty list.

Example :

{prompt_examples}

Text: 
{text}

Answer:
""" 
    
        # Construct request
        payload = {
            "prompt": prompt,
            "model" : 'command',
            "temperature" : 0,
            "max_tokens" : 200
        }     
        response = requests.post(url, json=payload, headers= self.headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)
        
        original_response = response.json()
        try:
            data = original_response.get('generations')[0]['text']
            items = json.loads(data)
        except (IndexError, KeyError, json.JSONDecodeError) as exc:
            raise ProviderException("An error occurred while parsing the response.") from exc
        
        standardized_response = CustomNamedEntityRecognitionDataClass(items=items)

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
