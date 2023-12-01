from typing import Dict, List
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import CustomNamedEntityRecognitionDataClass, GenerationDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.utils.exception import ProviderException
import boto3
import json

class AnthropicApi(
    ProviderInterface,
    TextInterface

):
    provider_name = "anthropic"
    
    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=self.api_settings["region_name"],
            aws_access_key_id=self.api_settings["aws_access_key_id"],
            aws_secret_access_key=self.api_settings["aws_secret_access_key"],
        )
    
    def _filter_entities_by_labels(self, items_data: dict, allowed_labels: list) -> dict:
        """
        Filters entities in the given 'items_data' based on the provided 'allowed_labels'.

        Args:
            items_data (dict): Input data containing a list of items with categories.
            allowed_labels (list): List of labels to retain in the filtered data.

        Returns:
            dict: Filtered data containing only items with categories in 'allowed_labels'.
        """
        filtered_data = {}
        filtered_data["items"] = [item for item in items_data["items"] if item["category"] in allowed_labels]
        return filtered_data
    
    def _construct_custom_ner_instruction(self, text: str, entities: list) -> str:
        return f"""
            You need to act like a named entity recognition model.
            Given the following list of entity types and a text, extract all the entities from the text that correspond to the entity types. Ensure that the same entity/category pair is not extracted twice. The output should be in the following JSON format:
            Expected JSON Output:
            {{
            "items": [
                {{"category": "<entity type from the entity list>", "entity": "<extracted entity from the text>"}},
                ...
            ]
            }}
            Your response should start immediately after the given text, without any introductory phrases.
            
            Entities: {entities}

            Text: {text}
            """
        
    def text__generation(
        self, 
        text: str,
        temperature: float, 
        max_tokens: int,
        model: str,) -> ResponseType[GenerationDataClass]:
        # Headers for the HTTP request
        accept_header = 'application/json'
        content_type_header = 'application/json'

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps({
            "prompt": f"\n\nHuman:{text}\n\nAssistant:",
            "temperature": temperature,
            "max_tokens_to_sample" : max_tokens
        })

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"{self.provider_name}.{model}",
            "accept": accept_header,
            "contentType": content_type_header
        }
        response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
        response_body = json.loads(response.get('body').read())
        generated_text = response_body['completion']
        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )

    def text__custom_named_entity_recognition(
        self,
        text: str,
        entities: List[str],
        examples: List[Dict]) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        # Headers for the HTTP request
        accept_header = 'application/json'
        content_type_header = 'application/json'
        prompt = self._construct_custom_ner_instruction(text, entities)
        
        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps({
            "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
            "temperature": 0,
            "max_tokens_to_sample" : 100000
        })

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"{self.provider_name}.claude-v2",
            "accept": accept_header,
            "contentType": content_type_header
        }
        response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
        response_body = json.loads(response.get('body').read())
        try:
            raw_items = json.loads(response_body['completion'])
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "Anthropic didn't return a valid JSON", code=500
            ) from exc
            
        items = self._filter_entities_by_labels(raw_items, entities)
        standardized_response = CustomNamedEntityRecognitionDataClass(
            items=items.get('items', [])
        )

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )