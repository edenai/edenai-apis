from abc import abstractmethod
from typing import Optional, List, Dict, Type, Union, Literal

from openai import BaseModel
import httpx

from edenai_apis.features.llm.chat.chat_dataclass import ChatDataClass


class LlmInterface:

    @abstractmethod
    def llm__chat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        modalities: Optional[List[Literal["text", "audio", "image"]]] = None,
        audio: Optional[Dict] = None,
        # openai v1.0+ new params
        response_format: Optional[
            Union[dict, Type[BaseModel]]
        ] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This should be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> ChatDataClass:
        """
        Generate responses in a multimodal conversation using a chatbot.
        Args:
            messages (List[Dict[str, str]]): A list of messages exchanged in the conversation.
            chatbot_global_action (Optional[str]): The global action or context for the chatbot.
            temperature (float, optional): Controls the randomness of the response generation.
            max_tokens (int, optional): The maximum number of tokens to generate for each response.
            model (Optional[str], optional): The name or identifier of the model.
            stop_sequences (Optional[List[str]], optional): A list of strings that, if encountered
                in the generated response, will stop generation.
            top_k (Optional[int], optional): Controls the diversity of the generated responses
                by limiting the number of tokens considered at each step.
            top_p (Optional[int], optional): Controls the diversity of the generated responses
                by selecting from the most probable tokens whose cumulative probability exceeds
                the given value.
            stream (bool, optional): Whether to enable streaming for generating responses.
            provider_params (dict, optional): Additional parameters specific to the provider
        """
        raise NotImplementedError

    @abstractmethod
    async def llm__achat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        modalities: Optional[List[Literal["text", "audio", "image"]]] = None,
        audio: Optional[Dict] = None,
        # openai v1.0+ new params
        response_format: Optional[
            Union[dict, Type[BaseModel]]
        ] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This should be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> ChatDataClass:
        """
        Async version of llm__chat method.
        Generate responses in a multimodal conversation using a chatbot.
        Args:
            messages (List[Dict[str, str]]): A list of messages exchanged in the conversation.
            chatbot_global_action (Optional[str]): The global action or context for the chatbot.
            temperature (float, optional): Controls the randomness of the response generation.
            max_tokens (int, optional): The maximum number of tokens to generate for each response.
            model (Optional[str], optional): The name or identifier of the model.
            stop_sequences (Optional[List[str]], optional): A list of strings that, if encountered
                in the generated response, will stop generation.
            top_k (Optional[int], optional): Controls the diversity of the generated responses
                by limiting the number of tokens considered at each step.
            top_p (Optional[int], optional): Controls the diversity of the generated responses
                by selecting from the most probable tokens whose cumulative probability exceeds
                the given value.
            stream (bool, optional): Whether to enable streaming for generating responses.
            provider_params (dict, optional): Additional parameters specific to the provider
        """
        raise NotImplementedError
