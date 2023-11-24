from typing import Dict
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.amazon.helpers import handle_amazon_call
import boto3
import json

class Ai21labsApi(ProviderInterface, TextInterface):
    provider_name = "ai21labs"

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

    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[GenerationDataClass]:
        # Headers for the HTTP request
        accept_header = 'application/json'
        content_type_header = 'application/json'

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps({
            "prompt": text,
            "maxTokens": max_tokens,
            "temperature": temperature
        })

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"ai21.{model}",
            "accept": accept_header,
            "contentType": content_type_header
        }
        response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
        response_body = json.loads(response.get('body').read())
        generated_text = response_body['completions'][0]['data']['text']
        
        # Calculate number of tokens : 
        prompt_tokens = len(response_body['prompt']['tokens'])
        completions_tokens = len(response_body['completions'][0]['data']['tokens'])
        response_body["usage"] = {
            "total_tokens" : prompt_tokens + completions_tokens
        }
        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )
