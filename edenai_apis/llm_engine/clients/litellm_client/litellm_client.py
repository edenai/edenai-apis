import logging
import time
from typing import List, Literal, Optional, Type, Union

import httpx
import litellm
from litellm import completion, completion_cost, embedding, moderation
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
from litellm.utils import get_supported_openai_params
from edenai_apis.llm_engine.types.litellm_model import LiteLLMModel
from llm_engine.clients.completion import CompletionClient
from llm_engine.exceptions.llm_engine_exceptions import CompletionClientError
from edenai_apis.utils.exception import ProviderException
from llm_engine.types.response_types import CustomStreamWrapperModel, ResponseModel

from pydantic import BaseModel
from litellm import register_model

logger = logging.getLogger(__name__)


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
        # self.model_capabilities = get_supported_openai_params(
        #     model=model_name, custom_llm_provider=provider_name
        # )

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
        stop=None,
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
        if model is not None:
            self.model_name = f"{self.provider_name}/{model}"

        call_params["model"] = self.model_name
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
        if stop is not None:
            call_params["stop"] = stop
        if stop_sequences is not None:
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
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            provider_start_time = time.time_ns()
            c_response = completion(**call_params, **kwargs)
            provider_end_time = time.time_ns()
            if stream:
                chunks = []
                for chunk in c_response:
                    if chunk is not None:
                        chunks.append(chunk)
                c_response = litellm.stream_chunk_builder(chunks, messages=messages)
            response = {
                **c_response.model_dump(),
                "cost": completion_cost(c_response),
                "provider_time": provider_end_time - provider_start_time,
            }
            return response
        except InternalServerError as e:
            logger.warning(f"There's an internal server error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.InternalServerError: ", ""),
                code=500,
            ) from e
        except APIError as e:
            logger.warning(f"There's an LiteLLM API error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.APIError: ", ""),
                code=e.status_code,
            ) from e
        except BadRequestError as e:
            logger.warning(
                f"There's a Bad Request error calling the provider {self.provider_name}: {e}"
            )
            raise ProviderException(
                message=e.message.replace("litellm.BadRequestError: ", ""),
                code=e.status_code,
            ) from e
        except AuthenticationError as e:
            logger.error(f"There's an authentication error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.AuthenticationError: ", ""),
                code=e.status_code,
            ) from e
        except Exception as e:
            logger.exception(e)
            raise ProviderException(message=str(e))

    def embedding(
        self,
        input=[],
        model: Optional[str] = None,
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
        call_params["model"] = self.model_name
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
        try:
            if drop_invalid_params == True:
                litellm.drop_params = True
            provider_start_time = time.time_ns()
            response = embedding(**call_params, **kwargs)
            response.provider_name = self.provider_name
            provider_end_time = time.time_ns()
            response.provider_time = provider_end_time - provider_start_time
            response.cost = completion_cost(response)
            return response
        except InternalServerError as e:
            logger.warning(f"There's an internal server error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.InternalServerError: ", ""),
                code=500,
            ) from e
        except APIError as e:
            logger.warning(f"There's an LiteLLM API error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.APIError: ", ""),
                code=e.status_code,
            ) from e
        except BadRequestError as e:
            logger.warning(
                f"There's a Bad Request error calling the provider {self.provider_name}: {e}"
            )
            raise ProviderException(
                message=e.message.replace("litellm.BadRequestError: ", ""),
                code=e.status_code,
            ) from e
        except AuthenticationError as e:
            logger.error(f"There's an authentication error: {e}")
            raise ProviderException(
                message=e.message.replace("litellm.AuthenticationError: ", ""),
                code=e.status_code,
            ) from e
        except Exception as e:
            logger.exception(e)
            raise ProviderException(message=str(e))

    def moderation(self, input: str, **kwargs):
        call_params = {}
        call_params["input"] = input
        try:
            # litellm.drop_params = True
            provider_start_time = time.time_ns()
            response = moderation(**call_params, **kwargs)
            response.provider_name = self.provider_name
            provider_end_time = time.time_ns()
            response.provider_time = provider_end_time - provider_start_time
            response.cost = 0
            return response
        except Exception as e:
            logging.error(f"There's an unexpected error: {e}")
            raise e

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
    
    def register_new_models(models: List[LiteLLMModel] = []):
        if models is None or len(models) == 0:
            return
        for model in models:
            try:
                cost = {}
                cost[model.model_name] = model.model_configuration.model_dump()
                register_model(cost)
            except Exception as e:
                logger.error(f"Error registering model {model.model_name}: {e}")

