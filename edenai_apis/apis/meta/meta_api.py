from typing import Dict, Literal, Optional, List, Union
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import GenerationDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatDataClass,
)
from edenai_apis.features.multimodal.chat.chat_dataclass import (
    ChatDataClass as ChatMultimodalDataClass,
    StreamChat as StreamChatMultimodal,
    ChatMessageDataClass as ChatMultimodalMessageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.llmengine.llm_engine import LLMEngine
import json
import boto3


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
        self.llm_client = LLMEngine(
            provider_name="bedrock",
            provider_config={
                "aws_access_key_id": self.api_settings["aws_access_key_id"],
                "aws_secret_access_key": self.api_settings["aws_secret_access_key"],
                "aws_region_name": self.api_settings["region_name"],
            },
        )

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
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
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
            text=text,
            chatbot_global_action=chatbot_global_action,
            previous_history=previous_history,
            temperature=temperature,
            max_tokens=max_tokens,
            model=f"{self.provider_name}.{model}",
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
            **kwargs,
        )
        return response
