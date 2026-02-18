from abc import abstractmethod
from typing import Any, Optional, List, Dict, Type, Union, Literal

from openai import BaseModel
import httpx

from edenai_apis.features.llm.chat.chat_dataclass import ChatDataClass
from edenai_apis.features.llm.responses.responses_dataclass import (
    DeleteResponseDataClass,
    ResponsesDataClass,
    StreamResponses,
)
from edenai_apis.features.llm.aresponses.aresponses_dataclass import StreamAResponses


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

    @abstractmethod
    def llm__responses(
        self,
        input: Union[str, List],
        model: Optional[str] = None,
        # Core Responses API params
        include: Optional[List] = None,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallel_tool_calls: Optional[bool] = None,
        prompt: Optional[dict] = None,
        previous_response_id: Optional[str] = None,
        reasoning: Optional[dict] = None,
        store: Optional[bool] = None,
        background: Optional[bool] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        text: Optional[dict] = None,
        text_format: Optional[Union[Type[BaseModel], dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        tools: Optional[List] = None,
        top_p: Optional[float] = None,
        truncation: Optional[Literal["auto", "disabled"]] = None,
        user: Optional[str] = None,
        service_tier: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        # Common params
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_invalid_params: bool = True,
        **kwargs,
    ) -> Union[ResponsesDataClass, StreamResponses]:
        """
        Generate a response using the OpenAI Responses API (or compatible provider).
        Args:
            input: A text string or list of input items (messages/content blocks).
            model: The model to use.
            include: Additional data to include in the response.
            instructions: System-level instructions for the model.
            max_output_tokens: Maximum number of output tokens to generate.
            metadata: Arbitrary metadata to attach to the response.
            parallel_tool_calls: Whether to allow parallel tool calls.
            prompt: A prompt object to use instead of raw input.
            previous_response_id: ID of a previous response for session continuity.
            reasoning: Reasoning configuration (e.g. effort level).
            store: Whether to store the response server-side.
            background: Whether to run the response in the background.
            stream: Whether to stream the response.
            temperature: Sampling temperature.
            text: Text formatting options (e.g. response format).
            text_format: Structured output schema for the text response.
            tool_choice: Controls how the model selects tools.
            tools: List of tools the model may use.
            top_p: Nucleus sampling parameter.
            truncation: Truncation strategy ('auto' or 'disabled').
            user: End-user identifier.
            service_tier: Service tier to use.
            safety_identifier: Safety identifier for content filtering.
            timeout: Request timeout.
            extra_headers: Additional HTTP headers.
            extra_query: Additional query parameters.
            extra_body: Additional body parameters.
            base_url: Override the base URL.
            api_version: API version string.
            api_key: Override the API key.
            custom_llm_provider: Override the LLM provider.
            drop_invalid_params: If True, unsupported parameters are dropped silently.
        """
        raise NotImplementedError

    @abstractmethod
    async def llm__aresponses(
        self,
        input: Union[str, List],
        model: Optional[str] = None,
        # Core Responses API params
        include: Optional[List] = None,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallel_tool_calls: Optional[bool] = None,
        prompt: Optional[dict] = None,
        previous_response_id: Optional[str] = None,
        reasoning: Optional[dict] = None,
        store: Optional[bool] = None,
        background: Optional[bool] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        text: Optional[dict] = None,
        text_format: Optional[Union[Type[BaseModel], dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        tools: Optional[List] = None,
        top_p: Optional[float] = None,
        truncation: Optional[Literal["auto", "disabled"]] = None,
        user: Optional[str] = None,
        service_tier: Optional[str] = None,
        safety_identifier: Optional[str] = None,
        # Common params
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_invalid_params: bool = True,
        **kwargs,
    ) -> Union[ResponsesDataClass, StreamAResponses]:
        """
        Async version of llm__responses.
        Generate a response using the OpenAI Responses API (or compatible provider).
        """
        raise NotImplementedError

    @abstractmethod
    def llm__get_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> ResponsesDataClass:
        """
        Retrieve a previously created response by its ID.
        Args:
            response_id: The ID of the response to retrieve.
            extra_headers: Additional HTTP headers.
            extra_query: Additional query parameters.
            extra_body: Additional body parameters.
            timeout: Request timeout.
            custom_llm_provider: Override the LLM provider.
        """
        raise NotImplementedError

    @abstractmethod
    async def llm__aget_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> ResponsesDataClass:
        """
        Async version of llm__get_responses.
        Retrieve a previously created response by its ID.
        """
        raise NotImplementedError

    @abstractmethod
    def llm__delete_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> DeleteResponseDataClass:
        """
        Delete a previously created response by its ID.
        Args:
            response_id: The ID of the response to delete.
            extra_headers: Additional HTTP headers.
            extra_query: Additional query parameters.
            extra_body: Additional body parameters.
            timeout: Request timeout.
            custom_llm_provider: Override the LLM provider.
        """
        raise NotImplementedError

    @abstractmethod
    async def llm__adelete_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ) -> DeleteResponseDataClass:
        """
        Async version of llm__delete_responses.
        Delete a previously created response by its ID.
        """
        raise NotImplementedError
