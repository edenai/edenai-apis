from typing import Dict, List, Literal, Optional, Type, Union

import httpx
from openai import BaseModel, OpenAI

from edenai_apis.features import ProviderInterface, LlmInterface, TextInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.llm.chat.chat_dataclass import (
    ChatDataClass,
    StreamChat as StreamChatCompletion,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.llmengine.types.response_types import ResponseModel


class IointelligenceApi(ProviderInterface, LlmInterface, TextInterface):
    provider_name = "iointelligence"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.intelligence.io.solutions/api/v1/"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

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
        completion_params = {"messages": messages, "model": model}
        if response_format is not None:
            completion_params["response_format"] = response_format
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        if temperature is not None:
            completion_params["temperature"] = temperature
        if tools is not None:
            completion_params["tools"] = tools
        if top_p is not None:
            completion_params["top_p"] = top_p
        if stream is not None:
            completion_params["stream"] = stream
        if frequency_penalty is not None:
            completion_params["frequency_penalty"] = frequency_penalty
        if logprobs is not None:
            completion_params["logprobs"] = logprobs
        if top_logprobs is not None:
            completion_params["top_logprobs"] = top_logprobs
        if n is not None:
            completion_params["n"] = n
        if presence_penalty is not None:
            completion_params["presence_penalty"] = presence_penalty
        if seed is not None:
            completion_params["seed"] = seed
        if stop is not None:
            completion_params["stop"] = stop
        if tool_choice is not None:
            completion_params["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            completion_params["parallel_tool_calls"] = parallel_tool_calls
        if user is not None:
            completion_params["user"] = user
        try:
            response = self.client.chat.completions.create(**completion_params)
            if stream:

                def generate_chunks():
                    for chunk in response:
                        if chunk is not None:
                            yield chunk.to_dict()
                            # yield ModelResponseStream.model(data)

                return StreamChatCompletion(stream=generate_chunks())
            else:
                response = response.to_dict()
                response_model = ResponseModel.model_validate(response)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        return response_model
