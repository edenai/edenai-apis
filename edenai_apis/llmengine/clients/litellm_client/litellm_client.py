import logging
import time
from typing import Any, Coroutine, Dict, List, Literal, Optional, Type, Union

import httpx
import litellm
from litellm import (
    acompletion,
    aembedding,  # async versions
    aimage_generation,
    amoderation,
    arerank,
    completion,
    completion_cost,
    embedding,
    image_generation,
    moderation,
    register_model,
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
from llmengine.clients.completion import CompletionClient
from llmengine.exceptions.error_handler import handle_litellm_exception
from llmengine.exceptions.llm_engine_exceptions import (
    CompletionClientError,
    RerankClientError,
)
from llmengine.types.response_types import (
    CustomStreamWrapperModel,
    RerankerResponse,
    ResponseModel,
)
from pydantic import BaseModel

from edenai_apis.llmengine.clients.reranker import RerankerClient
from edenai_apis.llmengine.types.litellm_model import LiteLLMModel
from edenai_apis.utils.exception import ProviderException

MOCK_RESPONSE = "Arrr, matey! What ye be seein’ in this here image is a grand pathway, made of wooden planks, weavin' its way through a lush and green landscape. The verdant grass sways in the gentle breeze, and the sky above be a brilliant blue, decorated with fluffy white clouds. Ye can spot trees and bushes on either side, makin' it a perfect setting for a stroll amongst nature. A peaceful place for a pirate at heart, aye!"

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
        # See if there's custom pricing (model_pricing for extended pricing, or legacy per-token pricing)
        model_pricing = kwargs.pop("model_pricing", None)
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

            if kwargs.pop("fake", False):
                kwargs["mock_response"] = MOCK_RESPONSE

            # Register custom model pricing in litellm's registry for extended pricing support
            if model_pricing:
                # Ensure required fields are set for litellm's cost calculation
                # litellm_provider is needed for provider matching in get_model_info
                # mode is needed for model type identification
                if "litellm_provider" not in model_pricing and self.provider_name:
                    model_pricing["litellm_provider"] = self.provider_name
                if "mode" not in model_pricing:
                    model_pricing["mode"] = "chat"
                # register_model merges with existing pricing via setdefault().update()
                # TODO: in the future we may want to find a better way to register the models instead of calling this each time
                # HACK: we register model_name.lower() as well to handle casses where litellm does a lookup with lower case model name (e.g. for together_ai models)
                # this is an issue in litellm that they need to fix, but this is a temporary workaround to make sure the custom pricing works for those models as well
                register_model(
                    {model_name: model_pricing, model_name.lower(): model_pricing}
                )
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
                # Use model_pricing via registry lookup, or fall back to legacy custom_cost_per_token
                if not model_pricing and len(custom_pricing.keys()) > 0:
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
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
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
                raise ProviderException(message=str(exc)) from exc

    async def aembedding(
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
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
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
            response = await aembedding(**call_params, **kwargs)
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
                raise ProviderException(message=str(exc)) from exc

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

    async def amoderation(self, input: str, **kwargs):
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
            response = (await amoderation(**call_params, **kwargs)).model_dump()
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

    async def aimage_generation(
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
            response = await aimage_generation(**call_params, **kwargs)
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
                raise ProviderException(message=str(exc)) from exc

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
                raise ProviderException(message=str(exc)) from exc

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
        # See if there's custom pricing (model_pricing for extended pricing, or legacy per-token pricing)
        model_pricing = kwargs.pop("model_pricing", None)
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

            if kwargs.pop("fake", False):
                kwargs["mock_response"] = MOCK_RESPONSE

            # Register custom model pricing in litellm's registry for extended pricing support
            if model_pricing:
                # Ensure required fields are set for litellm's cost calculation
                # litellm_provider is needed for provider matching in get_model_info
                # mode is needed for model type identification
                if "litellm_provider" not in model_pricing and self.provider_name:
                    model_pricing["litellm_provider"] = self.provider_name
                if "mode" not in model_pricing:
                    model_pricing["mode"] = "chat"
                # register_model merges with existing pricing via setdefault().update()
                # TODO: in the future we may want to find a better way to register the models instead of calling this each time
                # HACK: we register model_name.lower() as well to handle casses where litellm does a lookup with lower case model name (e.g. for together_ai models)
                # this is an issue in litellm that they need to fix, but this is a temporary workaround to make sure the custom pricing works for those models as well
                register_model(
                    {model_name: model_pricing, model_name.lower(): model_pricing}
                )

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
                    "call_type": "acompletion",
                }
                # Use model_pricing via registry lookup, or fall back to legacy custom_cost_per_token
                if not model_pricing and len(custom_pricing.keys()) > 0:
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

    def responses(
        self,
        input=None,
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
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_invalid_params: bool = True,
        **kwargs,
    ):
        if input is None:
            raise CompletionClientError("In responses, the input cannot be empty")
        call_params = {}
        model_name = f"{self.provider_name}/{model}"
        if self.provider_name is None:
            model_name = model
        call_params["model"] = model_name
        call_params["input"] = input
        if include is not None:
            call_params["include"] = include
        if instructions is not None:
            call_params["instructions"] = instructions
        if max_output_tokens is not None:
            call_params["max_output_tokens"] = max_output_tokens
        if metadata is not None:
            call_params["metadata"] = metadata
        if parallel_tool_calls is not None:
            call_params["parallel_tool_calls"] = parallel_tool_calls
        if prompt is not None:
            call_params["prompt"] = prompt
        if previous_response_id is not None:
            call_params["previous_response_id"] = previous_response_id
        if reasoning is not None:
            call_params["reasoning"] = reasoning
        if store is not None:
            call_params["store"] = store
        if background is not None:
            call_params["background"] = background
        if stream is not None:
            call_params["stream"] = stream
        if temperature is not None:
            call_params["temperature"] = temperature
        if text is not None:
            call_params["text"] = text
        if text_format is not None:
            call_params["text_format"] = text_format
        if tool_choice is not None:
            call_params["tool_choice"] = tool_choice
        if tools is not None:
            call_params["tools"] = tools
        if top_p is not None:
            call_params["top_p"] = top_p
        if truncation is not None:
            call_params["truncation"] = truncation
        if user is not None:
            call_params["user"] = user
        if service_tier is not None:
            call_params["service_tier"] = service_tier
        if safety_identifier is not None:
            call_params["safety_identifier"] = safety_identifier
        if timeout is not None:
            call_params["timeout"] = timeout
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if base_url is not None:
            call_params["base_url"] = base_url
        if api_version is not None:
            call_params["api_version"] = api_version
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        # See if there's custom pricing (model_pricing for extended pricing, or legacy per-token pricing)
        model_pricing = kwargs.pop("model_pricing", None)
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs.pop("input_cost_per_token")
            custom_pricing["output_cost_per_token"] = kwargs.pop(
                "output_cost_per_token"
            )
        try:
            if drop_invalid_params:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)

            if kwargs.pop("fake", False):
                kwargs["mock_response"] = MOCK_RESPONSE

            # Register custom model pricing in litellm's registry for extended pricing support
            if model_pricing:
                # Ensure required fields are set for litellm's cost calculation
                # litellm_provider is needed for provider matching in get_model_info
                # mode is needed for model type identification
                if "litellm_provider" not in model_pricing and self.provider_name:
                    model_pricing["litellm_provider"] = self.provider_name
                if "mode" not in model_pricing:
                    model_pricing["mode"] = "chat"
                # register_model merges with existing pricing via setdefault().update()
                # TODO: in the future we may want to find a better way to register the models instead of calling this each time
                # HACK: we register model_name.lower() as well to handle casses where litellm does a lookup with lower case model name (e.g. for together_ai models)
                # this is an issue in litellm that they need to fix, but this is a temporary workaround to make sure the custom pricing works for those models as well
                register_model(
                    {model_name: model_pricing, model_name.lower(): model_pricing}
                )

            provider_start_time = time.time_ns()
            r_response = litellm.responses(**call_params, **kwargs)
            provider_end_time = time.time_ns()
            if stream:

                def generate_chunks():
                    for chunk in r_response:
                        if chunk is not None:
                            yield chunk

                return generate_chunks()
            else:
                cost_calc_params = {
                    "completion_response": r_response,
                    "call_type": "responses",
                }
                # Use model_pricing via registry lookup, or fall back to legacy custom_cost_per_token
                if not model_pricing and len(custom_pricing.keys()) > 0:
                    cost_calc_params["custom_cost_per_token"] = custom_pricing
                response = {
                    **r_response.model_dump(),
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

    async def aresponses(
        self,
        input=None,
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
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_invalid_params: bool = True,
        **kwargs,
    ):
        if input is None:
            raise CompletionClientError("In aresponses, the input cannot be empty")
        call_params = {}
        model_name = f"{self.provider_name}/{model}"
        if self.provider_name is None:
            model_name = model
        call_params["model"] = model_name
        call_params["input"] = input
        if include is not None:
            call_params["include"] = include
        if instructions is not None:
            call_params["instructions"] = instructions
        if max_output_tokens is not None:
            call_params["max_output_tokens"] = max_output_tokens
        if metadata is not None:
            call_params["metadata"] = metadata
        if parallel_tool_calls is not None:
            call_params["parallel_tool_calls"] = parallel_tool_calls
        if prompt is not None:
            call_params["prompt"] = prompt
        if previous_response_id is not None:
            call_params["previous_response_id"] = previous_response_id
        if reasoning is not None:
            call_params["reasoning"] = reasoning
        if store is not None:
            call_params["store"] = store
        if background is not None:
            call_params["background"] = background
        if stream is not None:
            call_params["stream"] = stream
        if temperature is not None:
            call_params["temperature"] = temperature
        if text is not None:
            call_params["text"] = text
        if text_format is not None:
            call_params["text_format"] = text_format
        if tool_choice is not None:
            call_params["tool_choice"] = tool_choice
        if tools is not None:
            call_params["tools"] = tools
        if top_p is not None:
            call_params["top_p"] = top_p
        if truncation is not None:
            call_params["truncation"] = truncation
        if user is not None:
            call_params["user"] = user
        if service_tier is not None:
            call_params["service_tier"] = service_tier
        if safety_identifier is not None:
            call_params["safety_identifier"] = safety_identifier
        if timeout is not None:
            call_params["timeout"] = timeout
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if base_url is not None:
            call_params["base_url"] = base_url
        if api_version is not None:
            call_params["api_version"] = api_version
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        # See if there's custom pricing (model_pricing for extended pricing, or legacy per-token pricing)
        model_pricing = kwargs.pop("model_pricing", None)
        custom_pricing = {}
        if kwargs.get("input_cost_per_token", None) and kwargs.get(
            "output_cost_per_token", None
        ):
            custom_pricing["input_cost_per_token"] = kwargs.pop("input_cost_per_token")
            custom_pricing["output_cost_per_token"] = kwargs.pop(
                "output_cost_per_token"
            )
        try:
            if drop_invalid_params:
                litellm.drop_params = True
            kwargs.pop("moderate_content", None)

            if kwargs.pop("fake", False):
                kwargs["mock_response"] = MOCK_RESPONSE

            # Register custom model pricing in litellm's registry for extended pricing support
            if model_pricing:
                # Ensure required fields are set for litellm's cost calculation
                # litellm_provider is needed for provider matching in get_model_info
                # mode is needed for model type identification
                if "litellm_provider" not in model_pricing and self.provider_name:
                    model_pricing["litellm_provider"] = self.provider_name
                if "mode" not in model_pricing:
                    model_pricing["mode"] = "chat"
                # register_model merges with existing pricing via setdefault().update()
                # TODO: in the future we may want to find a better way to register the models instead of calling this each time
                # HACK: we register model_name.lower() as well to handle casses where litellm does a lookup with lower case model name (e.g. for together_ai models)
                # this is an issue in litellm that they need to fix, but this is a temporary workaround to make sure the custom pricing works for those models as well
                register_model(
                    {model_name: model_pricing, model_name.lower(): model_pricing}
                )

            provider_start_time = time.time_ns()
            r_response = await litellm.aresponses(**call_params, **kwargs)
            provider_end_time = time.time_ns()
            if stream:

                async def generate_chunks():
                    async for chunk in r_response:
                        if chunk is not None:
                            yield chunk

                return generate_chunks()
            else:
                cost_calc_params = {
                    "completion_response": r_response,
                    "call_type": "aresponses",
                }
                # Use model_pricing via registry lookup, or fall back to legacy custom_cost_per_token
                if not model_pricing and len(custom_pricing.keys()) > 0:
                    cost_calc_params["custom_cost_per_token"] = custom_pricing
                response = {
                    **r_response.model_dump(),
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


    def get_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        call_params = {"response_id": response_id}
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if timeout is not None:
            call_params["timeout"] = timeout
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        call_params.update(self.provider_config)
        try:
            response = litellm.get_responses(**call_params, **kwargs)
            return response.model_dump()
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError, APIConnectionError, APIResponseValidationError,
                    AuthenticationError, BadRequestError, NotFoundError,
                    RateLimitError, ServiceUnavailableError, ContentPolicyViolationError,
                    Timeout, UnprocessableEntityError, JSONSchemaValidationError,
                    UnsupportedParamsError, ContextWindowExceededError, InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc)) from exc

    async def aget_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        call_params = {"response_id": response_id}
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if timeout is not None:
            call_params["timeout"] = timeout
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        call_params.update(self.provider_config)
        try:
            response = await litellm.aget_responses(**call_params, **kwargs)
            return response.model_dump()
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError, APIConnectionError, APIResponseValidationError,
                    AuthenticationError, BadRequestError, NotFoundError,
                    RateLimitError, ServiceUnavailableError, ContentPolicyViolationError,
                    Timeout, UnprocessableEntityError, JSONSchemaValidationError,
                    UnsupportedParamsError, ContextWindowExceededError, InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc)) from exc

    def delete_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        call_params = {"response_id": response_id}
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if timeout is not None:
            call_params["timeout"] = timeout
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        call_params.update(self.provider_config)
        try:
            response = litellm.delete_responses(**call_params, **kwargs)
            return response.model_dump()
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError, APIConnectionError, APIResponseValidationError,
                    AuthenticationError, BadRequestError, NotFoundError,
                    RateLimitError, ServiceUnavailableError, ContentPolicyViolationError,
                    Timeout, UnprocessableEntityError, JSONSchemaValidationError,
                    UnsupportedParamsError, ContextWindowExceededError, InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc)) from exc

    async def adelete_responses(
        self,
        response_id: str,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        call_params = {"response_id": response_id}
        if extra_headers is not None:
            call_params["extra_headers"] = extra_headers
        if extra_query is not None:
            call_params["extra_query"] = extra_query
        if extra_body is not None:
            call_params["extra_body"] = extra_body
        if timeout is not None:
            call_params["timeout"] = timeout
        if custom_llm_provider is not None:
            call_params["custom_llm_provider"] = custom_llm_provider
        call_params.update(self.provider_config)
        try:
            response = await litellm.adelete_responses(**call_params, **kwargs)
            return response.model_dump()
        except Exception as exc:
            if isinstance(
                exc,
                (
                    APIError, APIConnectionError, APIResponseValidationError,
                    AuthenticationError, BadRequestError, NotFoundError,
                    RateLimitError, ServiceUnavailableError, ContentPolicyViolationError,
                    Timeout, UnprocessableEntityError, JSONSchemaValidationError,
                    UnsupportedParamsError, ContextWindowExceededError, InternalServerError,
                ),
            ):
                raise handle_litellm_exception(exc) from exc
            else:
                raise ProviderException(str(exc)) from exc


class LiteLLMRerankClient(RerankerClient):
    CLIENT_NAME = "rerank_litellm"

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

    async def arerank(
        self,
        model: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[
            Literal["cohere", "together_ai", "azure_ai", "infinity", "litellm_proxy"]
        ] = None,
        api_key: str = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
        **kwargs,
    ) -> Coroutine[Any, Any, RerankerResponse]:
        call_params = {}
        model_name = f"{self.provider_name}/{model}"
        if self.provider_name is None:
            model_name = model
        if not model_name:
            raise RerankClientError(
                message="You must provide the model name", status_code=400
            )
        call_params["model"] = model_name
        if not query:
            raise RerankClientError(message="You must provide a query", status_code=400)
        call_params["query"] = query
        if not documents:
            raise RerankClientError(
                message="You must provide a list of documents to search",
                status_code=400,
            )
        call_params["documents"] = documents
        ### Optional params...
        if api_key:
            call_params["api_key"] = api_key
        if top_n:
            call_params["top_n"] = top_n
        if rank_fields:
            call_params["rank_fields"] = rank_fields
        if return_documents is not None:
            call_params["return_documents"] = return_documents
        if max_chunks_per_doc:
            call_params["max_chunks_per_doc"] = max_chunks_per_doc
        if max_tokens_per_doc:
            call_params["max_tokens_per_doc"] = max_tokens_per_doc

        try:
            response = await arerank(**call_params, **kwargs)
            return RerankerResponse(**(response.model_dump()))
        except Exception as ex:
            if isinstance(
                ex,
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
                raise handle_litellm_exception(ex) from ex
            else:
                raise ProviderException(str(ex)) from ex
