import logging
import time
import uuid
from typing import Any, List, Literal, Optional, Type, Union

import httpx
from llm_engine.types.response_types import CustomStreamWrapperModel, ResponseModel
from llm_engine.clients import LLM_COMPLETION_CLIENTS
from llm_engine.clients.completion import CompletionClient
from llm_engine.exceptions.llm_engine_exceptions import LLMEngineError
# from llm_engine.types.response_types import CustomStreamWrapperModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Instantiate the engine for doing calls to LLMs
    """

    def __init__(
        self,
        provider_name: str,
        model: Optional[str] = None,
        client_name: Optional[str] = None,
        application_name: str = uuid.uuid4(),
        **kwargs,
    ) -> None:
        # Verify that the model is not empty
        if provider_name == "":
            raise LLMEngineError("The model name cannot be empty")
        # Set the user
        self.model = model
        self.provider_name = provider_name
        self.application_name = str(application_name)
        if client_name is None:
            client_name = next(iter(LLM_COMPLETION_CLIENTS))
        # TODO change the completion client to behave in the same way
        self.completion_client: CompletionClient = LLM_COMPLETION_CLIENTS[client_name](
            model_name=model, provider_name=self.provider_name
        )

    def completion(
        self,
        model: Optional[str] = None,
        messages: List = [],
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        stop_sequences=None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        # openai v1.0+ new params
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This shouold be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        # Optional parameters
        user: Optional[str] = None,
        **kwargs,
    ) -> Union[ResponseModel, CustomStreamWrapperModel]:
        if messages is None:
            raise LLMEngineError("In completion, the messages cannot be empty")
        call_params: dict[str, Any] = {"messages": messages}
        if model is not None:
            call_params["model"] = model
        if timeout is not None:
            call_params["timeout"] = timeout
        if temperature is not None:
            call_params["temperature"] = temperature
        if top_p is not None:
            call_params["top_p"] = top_p
        if n is not None:
            call_params["n"] = n
        if stream is not None:
            call_params["stream"] = stream
        if stream_options is not None:
            call_params["stream_options"] = stream_options
        if stop is not None:
            call_params["stop"] = stop
        if stop_sequences is not None:
            if isinstance(stop_sequences, list) and len(stop_sequences) > 4:
                logger.warning(
                    f"stop_sequences contains {len(stop_sequences)}. I'll take the first 4 and drop the other. Sorry"
                )
                stop_sequences = stop_sequences[:4]
            call_params["stop"] = stop_sequences
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            call_params["logit_bias"] = logit_bias
        if response_format is not None:
            call_params["response_format"] = response_format
        if seed is not None:
            call_params["seed"] = seed
        if tools is not None:
            call_params["tools"] = tools
        if tool_choice is not None:
            call_params["tool_choice"] = tool_choice
        if logprobs is not None:
            call_params["logprobs"] = logprobs
        if top_logprobs is not None:
            call_params["top_logprobs"] = top_logprobs
        if parallel_tool_calls is not None:
            call_params["parallel_tool_calls"] = parallel_tool_calls
        if deployment_id is not None:
            call_params["deployment_id"] = deployment_id
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if functions is not None:
            call_params["functions"] = functions
        if function_call is not None:
            call_params["function_call"] = function_call
        if base_url is not None:
            call_params["base_url"] = base_url
        if api_version is not None:
            call_params["api_version"] = api_version
        if model_list is not None:
            call_params["model_list"] = model_list
        self.history_input_log = call_params.copy()
        if api_key is not None:
            call_params["api_key"] = api_key
        # Coherence check
        if "tool_choice" in call_params and "tools" not in call_params:
            logger.warning("You specified a tool_choice but no tools. I'll ignore the tool_choice.")
            call_params["tool_choice"] = None

        # send unique organization identifier to provider
        if user is not None:
            call_params["user"] = user

        # Verify that there's not a mocked response...
        if "mocked_response" in kwargs:
            call_params["mocked_response"] = kwargs["mocked_response"]
        if not stream:
            edenai_start_time = time.time_ns()
            response = self.completion_client.completion(**call_params, **kwargs)
            edenai_end_time = time.time_ns()
            response["edenai_time"] = edenai_end_time - edenai_start_time - response["provider_time"]
            response = ResponseModel.model_validate(response)
            return response
        else:
            stream_response = self.completion_client.completion(**call_params, **kwargs)
            return stream_response

    def embedding(
        self,
        input=[],
        # Optional params
        dimensions: Optional[int] = None,
        timeout=600,  # default to 10 minutes
        # set api_base, api_version, api_key
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        api_type: Optional[str] = None,
        caching: bool = False,
        encoding_format: Literal["float", "base64"] = "float",
        drop_invalid_params: bool = False,
        **kwargs,
    ):
        if input is None:
            raise LLMEngineError("In embedding, the input cannot be empty")
        call_params = {"input": input}
        if dimensions is not None:
            call_params["dimensions"] = dimensions
        if timeout is not None:
            call_params["timeout"] = timeout
        if api_base is not None:
            call_params["api_base"] = api_base
        if api_version is not None:
            call_params["api_version"] = api_version
        if api_key is not None:
            call_params["api_key"] = api_key
        if api_type is not None:
            call_params["api_type"] = api_type
        if caching is not None:
            call_params["caching"] = caching
        if encoding_format is not None:
            call_params["encoding_format"] = encoding_format
        call_params["drop_invalid_params"] = drop_invalid_params
        edenai_start_time = time.time_ns()
        response = self.completion_client.embedding(**call_params, **kwargs)
        edenai_end_time = time.time_ns()
        response.edenai_time = edenai_end_time - edenai_start_time
        return response

    def moderation(self, input: str, api_key: Optional[str] = None, **kwargs):
        call_params = {}
        call_params["api_key"] = api_key
        call_params["input"] = input
        edenai_start_time = time.time_ns()
        response = self.completion_client.moderation(**call_params, **kwargs)
        edenai_end_time = time.time_ns()
        response.edenai_time = edenai_end_time - edenai_start_time
        return response

    def __str__(self) -> str:
        return f"LLMEngine(model_name={self.model_name}, provider_name={self.provider_name}, completion_client={self.completion_client})"

    def get_input_log(self) -> dict:
        return self.history_input_log
