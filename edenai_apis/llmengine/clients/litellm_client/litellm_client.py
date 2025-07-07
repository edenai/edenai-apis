import logging
import time
from typing import List, Literal, Optional, Type, Union

import httpx
import litellm
from litellm import (
    completion,
    completion_cost,
    embedding,
    image_generation,
    moderation,
    register_model,
    response_cost_calculator,
    acompletion,
)
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    JSONSchemaValidationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)
from litellm.utils import get_supported_openai_params
from llmengine.clients.completion import CompletionClient
from llmengine.exceptions.error_handler import handle_litellm_exception
from llmengine.exceptions.llm_engine_exceptions import CompletionClientError
from llmengine.types.response_types import CustomStreamWrapperModel, ResponseModel
from pydantic import BaseModel

from edenai_apis.llmengine.types.litellm_model import LiteLLMModel
from edenai_apis.utils.exception import ProviderException

logger = logging.getLogger(__name__)

litellm.modify_params = True


class LiteLLMCompletionClient(CompletionClient):

    CLIENT_NAME = "litellm"

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_name: str = None,
        provider_config: dict = {},
    ):
        super().__init__(
            model_name=model_name,
            provider_name=provider_name,
            provider_config=provider_config,
        )

    def completion(
        self,
        messages: List = [],
        model: Optional[str] = None,  # TODO This should be some by default thing
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
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = False,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> Union[ResponseModel, CustomStreamWrapperModel]:
        if messages is None:
            raise CompletionClientError("In completion, the messages cannot be empty")
        call_params = {}
        model_name = f"{self.provider_name}/{model}"
        if self.provider_name is None:
            model_name = model
        call_params["model"] = model_name
        call_params["messages"] = messages
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
        if stop is None:
            stop = kwargs.pop("stop", None)
        if stop is not None and len(stop) != 0:
            call_params["stop"] = stop
        if stop_sequences is not None and len(stop_sequences) > 0:
            call_params["stop"] = stop_sequences
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            call_params["logit_bias"] = logit_bias
        # Process response format
        response_format = self._process_response_format(response_format)
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
            logger.warning(
                "Functions are going to be deprecated in the near future (at least by OpenAI)"
            )
            call_params["functions"] = functions
        if function_call is not None:
            logger.warning(
                "Functions are going to be deprecated in the near future (at least by OpenAI)"
            )
            call_params["function_call"] = function_call
        if base_url is not None:
            call_params["base_url"] = base_url
        if api_version is not None:
            call_params["api_version"] = api_version
        if model_list is not None:
            call_params["model_list"] = model_list
        if user is not None:
            call_params["user"] = user
        # See if there's a custom pricing here
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs.pop("input_cost_per_token")
            custom_pricing["output_cost_per_token"] = kwargs.pop(
                "output_cost_per_token"
            )
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)
            provider_start_time = time.time_ns()
            c_response = completion(**call_params, **kwargs)
            provider_end_time = time.time_ns()
            if stream:

                def generate_chunks():
                    for chunk in c_response:
                        if chunk is not None:
                            yield chunk

                return generate_chunks()
            else:
                cost_calc_params = {
                    "completion_response": c_response,
                    "call_type": "completion",
                }
                if len(custom_pricing.keys()) > 0:
                    cost_calc_params["custom_cost_per_token"] = custom_pricing
                response = {
                    **c_response.model_dump(),
                    "cost": completion_cost(**cost_calc_params),
                    "provider_time": provider_end_time - provider_start_time,
                }
                return response

        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError,
                    APIConnectionError,
                    APIResponseValidationError,
                    AuthenticationError,
                    BadRequestError,
                    NotFoundError,
                    RateLimitError,
                    ServiceUnavailableError,
                    ContentPolicyViolationError,
                    Timeout,
                    UnprocessableEntityError,
                    JSONSchemaValidationError,
                    UnsupportedParamsError,
                    ContextWindowExceededError,
                    InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc))

    def embedding(
        self,
        input=[],
        model: Optional[str] = None,
        provider_model_name: Optional[str] = None,
        # Optional params
        dimensions: Optional[int] = None,
        timeout=600,  # default to 10 minutes
        # set api_base, api_version, api_key
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_type: Optional[str] = None,
        caching: bool = False,
        drop_invalid_params: bool = False,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        encoding_format: Literal["float", "base64"] = "float",
        **kwargs,
    ):
        call_params = {}
        call_params["input"] = input if isinstance(input, list) else [input]
        if model is not None:
            self.model_name = model
        call_params["model"] = f"{self.provider_name}/{model}"
        if provider_model_name:
            call_params["model"] = provider_model_name
        call_params["timeout"] = timeout
        if dimensions is not None:
            call_params["dimensions"] = dimensions
        if api_base is not None:
            call_params["api_base"] = api_base
        if api_version is not None:
            call_params["api_version"] = api_version
        if api_type is not None:
            call_params["api_type"] = api_type
        if caching is not None:
            call_params["caching"] = caching
        if encoding_format is not None:
            call_params["encoding_format"] = encoding_format
        # See if there's a custom pricing here
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs["input_cost_per_token"]
            custom_pricing["output_cost_per_token"] = kwargs["output_cost_per_token"]
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)
            provider_start_time = time.time_ns()
            response = embedding(**call_params, **kwargs)
            response.provider_name = self.provider_name
            provider_end_time = time.time_ns()
            response.provider_time = provider_end_time - provider_start_time
            cost_calc_params = {
                "completion_response": response,
                "call_type": "embedding",
            }
            if len(custom_pricing.keys()) > 0:
                cost_calc_params["custom_cost_per_token"] = custom_pricing
            response.cost = completion_cost(**cost_calc_params)
            return response
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError,
                    APIConnectionError,
                    APIResponseValidationError,
                    AuthenticationError,
                    BadRequestError,
                    NotFoundError,
                    RateLimitError,
                    ServiceUnavailableError,
                    ContentPolicyViolationError,
                    Timeout,
                    UnprocessableEntityError,
                    JSONSchemaValidationError,
                    UnsupportedParamsError,
                    ContextWindowExceededError,
                    InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(message=str(exc))

    def moderation(self, input: str, **kwargs):
        call_params = {}
        call_params["input"] = input
        # See if there's a custom pricing here
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs["input_cost_per_token"]
            custom_pricing["output_cost_per_token"] = kwargs["output_cost_per_token"]
        try:
            # litellm.drop_params = True
            kwargs.pop("moderate_content", None)
            provider_start_time = time.time_ns()
            response = moderation(**call_params, **kwargs).model_dump()
            provider_end_time = time.time_ns()

            cost_calc_params = {
                "completion_response": response,
                "call_type": "moderation",
            }
            if len(custom_pricing.keys()) > 0:
                cost_calc_params["custom_cost_per_token"] = custom_pricing

            response["cost"] = completion_cost(**cost_calc_params)
            response["provider_time"] = provider_end_time - provider_start_time
            return response
        except Exception as e:
            logging.error(f"There's an unexpected error: {e}")
            raise e

    def image_generation(
        self,
        prompt: str,
        n: Optional[int] = None,
        model: Optional[str] = None,
        quality: Optional[str] = None,
        response_format: Optional[str] = None,
        size: Optional[str] = None,
        style: Optional[str] = None,
        user: Optional[str] = None,
        timeout=600,  # default to 10 minutes
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider=None,
        drop_invalid_params: Optional[bool] = True,
        **kwargs,
    ):
        call_params = {}
        call_params["prompt"] = prompt
        if n is not None:
            call_params["n"] = n
        if quality is not None:
            call_params["quality"] = quality
        if response_format is not None:
            call_params["response_format"] = response_format
        if size is not None:
            call_params["size"] = size
        if style is not None:
            call_params["style"] = style
        if user is not None:
            call_params["user"] = user
        model_name = f"{self.provider_name}/{model}"
        call_params["model"] = model_name
        call_params["timeout"] = timeout
        if api_base is not None:
            call_params["api_base"] = api_base
        if api_version is not None:
            call_params["api_version"] = api_version
        if api_key is not None:
            call_params["api_key"] = api_key
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs["input_cost_per_token"]
            custom_pricing["output_cost_per_token"] = kwargs["output_cost_per_token"]
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)
            provider_start_time = time.time_ns()
            response = image_generation(**call_params, **kwargs)
            response.provider_name = self.provider_name
            provider_end_time = time.time_ns()
            cost_calc_params = {
                "completion_response": response,
                "call_type": "image_generation",
                "model": model_name,
            }
            if len(custom_pricing.keys()) > 0:
                cost_calc_params["custom_cost_per_token"] = custom_pricing

            response.provider_time = provider_end_time - provider_start_time
            response.cost = completion_cost(**cost_calc_params)
            return response
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError,
                    APIConnectionError,
                    APIResponseValidationError,
                    AuthenticationError,
                    BadRequestError,
                    NotFoundError,
                    RateLimitError,
                    ServiceUnavailableError,
                    ContentPolicyViolationError,
                    Timeout,
                    UnprocessableEntityError,
                    JSONSchemaValidationError,
                    UnsupportedParamsError,
                    ContextWindowExceededError,
                    InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(message=str(exc))

    def _process_response_format(self, input_response_format: any) -> any:
        """
        This is to make sure that the response format is well formated before passing it to LiteLLM:
        if the response format contains 'type':'json_schema' but without especifying the 'json_schema',
        we may need to convert the object to 'type':'json_object"""
        if input_response_format is None:
            return None
        if isinstance(input_response_format, dict):
            keys = input_response_format.keys()
            if len(keys) == 1 and "type" in keys:
                if input_response_format["type"] == "json_schema":
                    input_response_format["type"] = "json_object"
            elif len(keys) == 0:
                input_response_format = None
        return input_response_format

    def register_new_models(self, models: List[LiteLLMModel] = []):
        if models is None or len(models) == 0:
            return
        for model in models:
            try:
                cost = {}
                cost[model.model_name] = model.model_configuration.model_dump()
                register_model(cost)
            except Exception as e:
                logger.error(f"Error registering model {model.model_name}: {e}")

    # async versions of the methods

    async def acompletion(
        self,
        messages: List = [],
        model: Optional[str] = None,  # TODO This should be some by default thing
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
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = False,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> Union[ResponseModel, CustomStreamWrapperModel]:
        if messages is None:
            raise CompletionClientError("In completion, the messages cannot be empty")
        call_params = {}
        model_name = f"{self.provider_name}/{model}"
        if self.provider_name is None:
            model_name = model
        call_params["model"] = model_name
        call_params["messages"] = messages
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
        if stop is None:
            stop = kwargs.pop("stop", None)
        if stop is not None and len(stop) != 0:
            call_params["stop"] = stop
        if stop_sequences is not None and len(stop_sequences) > 0:
            call_params["stop"] = stop_sequences
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            call_params["logit_bias"] = logit_bias
        # Process response format
        response_format = self._process_response_format(response_format)
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
            logger.warning(
                "Functions are going to be deprecated in the near future (at least by OpenAI)"
            )
            call_params["functions"] = functions
        if function_call is not None:
            logger.warning(
                "Functions are going to be deprecated in the near future (at least by OpenAI)"
            )
            call_params["function_call"] = function_call
        if base_url is not None:
            call_params["base_url"] = base_url
        if api_version is not None:
            call_params["api_version"] = api_version
        if model_list is not None:
            call_params["model_list"] = model_list
        if user is not None:
            call_params["user"] = user
        # See if there's a custom pricing here
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs.pop("input_cost_per_token")
            custom_pricing["output_cost_per_token"] = kwargs.pop(
                "output_cost_per_token"
            )
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)
            provider_start_time = time.time_ns()
            c_response = await acompletion(**call_params, **kwargs)
            provider_end_time = time.time_ns()
            if stream:

                async def generate_chunks():
                    async for chunk in c_response:
                        if chunk is not None:
                            yield chunk

                return generate_chunks()
            else:
                cost_calc_params = {
                    "completion_response": c_response,
                    "call_type": "completion",
                }
                if len(custom_pricing.keys()) > 0:
                    cost_calc_params["custom_cost_per_token"] = custom_pricing
                response = {
                    **c_response.model_dump(),
                    "cost": completion_cost(**cost_calc_params),
                    "provider_time": provider_end_time - provider_start_time,
                }
                return response

        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError,
                    APIConnectionError,
                    APIResponseValidationError,
                    AuthenticationError,
                    BadRequestError,
                    NotFoundError,
                    RateLimitError,
                    ServiceUnavailableError,
                    ContentPolicyViolationError,
                    Timeout,
                    UnprocessableEntityError,
                    JSONSchemaValidationError,
                    UnsupportedParamsError,
                    ContextWindowExceededError,
                    InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc)) from exc
