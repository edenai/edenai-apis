from typing import Dict, Literal, Optional, List, Union
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ChatDataClass,
    ChatMessageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException
from edenai_apis.apis.amazon.helpers import handle_amazon_call
import boto3
import json


class MetaApi(ProviderInterface, TextInterface):
    provider_name = "meta"

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
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request, containing text, maxTokens, and temperature
        formatted_prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {text}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        request_body = json.dumps(
            {
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
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
        generated_text = response_body["generation"]

        # Calculate number of tokens :
        response_body["usage"] = {
            "total_tokens": response_body["prompt_token_count"]
            + response_body["generation_token_count"]
        }

        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str] = None,
        previous_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stream: bool = False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        """
        For the new llama3 models the prompting format is :
        <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>{user_message_1}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>{assistant_message_1}<|eot_id|>
            ...
            <|start_header_id|>user<|end_header_id|>{current_text}<|eot_id|>
        """

        if any([available_tools, tool_results]):
            raise ProviderException("This provider does not support the use of tools")

        prompt = "<|begin_of_text|>\n"

        if chatbot_global_action:
            prompt += f"<|start_header_id|>system<|end_header_id|>{chatbot_global_action}<|eot_id|>\n"

        for msg in previous_history or []:
            role = msg["role"]
            message = msg["message"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>{message}<|eot_id|>\n"

        prompt += f"<|start_header_id|>user<|end_header_id|>{text}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Headers for the HTTP request
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps(
            {"prompt": prompt, "max_gen_len": max_tokens, "temperature": temperature}
        )

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"{self.provider_name}.{model}",
            "accept": accept_header,
            "contentType": content_type_header,
        }
        if stream is False:
            response = handle_amazon_call(self.bedrock.invoke_model, **request_params)
            response_body = json.loads(response.get("body").read())
            generated_text = response_body["generation"]

            # Build a list of ChatMessageDataClass objects for the conversation history
            message = [
                ChatMessageDataClass(role="user", message=text),
                ChatMessageDataClass(role="assistant", message=generated_text),
            ]

            # Build the standardized response
            standardized_response = ChatDataClass(
                generated_text=generated_text, message=message
            )

            # Calculate number of tokens :
            response_body["usage"] = {
                "total_tokens": response_body["prompt_token_count"]
                + response_body["generation_token_count"]
            }

            return ResponseType[ChatDataClass](
                original_response=response_body,
                standardized_response=standardized_response,
            )
        else:
            response = handle_amazon_call(
                self.bedrock.invoke_model_with_response_stream, **request_params
            )
            stream = (
                ChatStreamResponse(
                    text=json.loads(event["chunk"]["bytes"])["generation"],
                    blocked=not json.loads(event["chunk"]["bytes"])["stop_reason"]
                    in (None, "stop"),
                    provider="meta",
                )
                for event in response.get("body")
                if event
            )

            return ResponseType[StreamChat](
                original_response=None, standardized_response=StreamChat(stream=stream)
            )
