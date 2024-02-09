from typing import Dict
import json
import boto3
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass, SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.amazon.helpers import handle_amazon_call


class AnthropicApi(ProviderInterface, TextInterface):
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

    def __anthropic_request(
        self, text: str, model: str, temperature: int = 0, max_tokens=10000
    ):
        # Headers for the HTTP request
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps(
            {
                "prompt": text,
                "temperature": temperature,
                "max_tokens_to_sample": max_tokens,
            }
        )
        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"{self.provider_name}.{model}",
            "accept": accept_header,
            "contentType": content_type_header,
        }
        response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
        response_body = json.loads(response.get("body").read())
        return response_body

    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[GenerationDataClass]:
        prompt = f"\n\nHuman:{text}\n\nAssistant:"
        response_body = self.__anthropic_request(
            text=prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
        generated_text = response_body["completion"]
        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str = None
    ) -> ResponseType[SummarizeDataClass]:
        prompt = f"""\n\nHuman: Given the following text, please provide a concise summary in the same language: text : {text}\n\nAssistant: 
        Summary:
        """
        original_response = self.__anthropic_request(
            text=prompt, model=model, max_tokens=100000
        )
        standardized_response = SummarizeDataClass(
            result=original_response.get("completion")
        )
        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
