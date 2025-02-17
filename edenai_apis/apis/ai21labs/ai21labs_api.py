from typing import Dict, Any, List, Optional, Union
import json
import requests
import boto3
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass, SummarizeDataClass
from edenai_apis.features.text.embeddings.embeddings_dataclass import (
    EmbeddingsDataClass,
    EmbeddingDataClass,
)
from edenai_apis.features.text.spell_check.spell_check_dataclass import (
    SpellCheckDataClass,
    SpellCheckItem,
    SuggestionItem,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.utils.exception import ProviderException


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
        self.base_url = "https://api.ai21.com/studio/v1"
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # TODO remove bedrock from text generation & replace it with Ai21labs API
    def __ai21labs_bedrock_request(
        self, text: str, model: str, temperature: int = 0, max_tokens: int = None
    ) -> Dict:
        # Headers for the HTTP request
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps(
            {"prompt": text, "maxTokens": max_tokens, "temperature": temperature}
        )

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"ai21.{model}",
            "accept": accept_header,
            "contentType": content_type_header,
        }
        response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
        response_body = json.loads(response.get("body").read())
        return response_body

    def __ai21labs_api_request(
        self, url: str, payload: Dict[str, Any]
    ) -> Union[Dict[str, Any], None]:
        """
        Make a request to the AI21 Labs API.
        Args:
            url (str): The URL of the API endpoint.
            payload (Dict[str, Any]): The payload to be sent with the request.

        Returns:
            Union[Dict[str, Any], None]: The JSON response from the API, or None if there's an error.
        """
        response = requests.post(
            f"{self.base_url}/{url}", json=payload, headers=self.headers
        )
        try:
            original_response = response.json()
            if response.status_code != 200:
                message_error = original_response.get("detail", "Unknown Error")
                raise ProviderException(message_error, code=response.status_code)
            return original_response
        except json.JSONDecodeError as exc:
            raise ProviderException(response.text, code=response.status_code) from exc

    def __calculate_generated_tokens(self, original_response: Dict) -> int:
        prompt_tokens = len(original_response["prompt"]["tokens"])
        completions_tokens = len(original_response["completions"][0]["data"]["tokens"])
        total_tokens = prompt_tokens + completions_tokens
        return total_tokens

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
    ) -> ResponseType[GenerationDataClass]:
        response_body = self.__ai21labs_bedrock_request(
            text=text, temperature=temperature, max_tokens=max_tokens, model=model
        )
        generated_text = response_body["completions"][0]["data"]["text"]

        total_tokens = self.__calculate_generated_tokens(response_body)
        response_body["usage"] = {"total_tokens": total_tokens}

        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None, **kwargs
    ) -> ResponseType[EmbeddingsDataClass]:
        payload = {"texts": texts}
        original_response = self.__ai21labs_api_request(url="embed", payload=payload)
        embeddings = original_response["results"]
        items = []
        for embedding in embeddings:
            items.append(EmbeddingDataClass(embedding=embedding["embedding"]))

        standardized_response = EmbeddingsDataClass(items=items)

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
