from edenai_apis.features import ProviderInterface, TextInterface, LlmInterface
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatStreamResponse
from typing import Dict, List, Literal, Optional, Union, Generator
from edenai_apis.llmengine.llm_engine import LLMEngine
from openai import BaseModel
from typing import Type
import httpx
import requests
import json


class PerplexityApi(ProviderInterface, TextInterface, LlmInterface):
    provider_name = "perplexityai"

    def __init__(self, api_keys: Dict = {}):
        self.url = "https://api.perplexity.ai"

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        self.api_key = self.api_settings["api_key"]

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

        self.llm_client = LLMEngine(
            provider_name="perplexity",
            provider_config={
                "api_key": self.api_key,
            },
        )

    @staticmethod
    def __text_to_json(
        lst_data: List[str],
    ) -> Generator[ChatStreamResponse, None, None]:
        length = len(lst_data)
        i = 0
        while i < length:
            if lst_data[i].startswith("data:"):
                lst_data[i] = lst_data[i].replace("data: ", "")
                lst_data[i] = lst_data[i].replace("\r", "")
                i += 1
            else:
                lst_data.pop(i)
            length = len(lst_data)
        for token in lst_data:
            jsonres = json.loads(token)
            if error := jsonres.get("error"):
                raise ProviderException(error.get("message"), error.get("code") or 400)
            yield ChatStreamResponse(
                text=jsonres["choices"][0]["delta"]["content"],
                blocked=False,
                provider="perplexityai",
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
        messages = []

        if any([available_tools, tool_results]):
            raise ProviderException("This provider does not support the use of tools")

        if chatbot_global_action:
            messages.append({"role": "system", "content": chatbot_global_action})

        if previous_history:
            for message in previous_history:
                messages.append(
                    {"role": message.get("role"), "content": message.get("message")},
                )

        messages.append({"role": "user", "content": text})
        url = f"{self.url}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)
        else:
            if not stream:
                try:
                    original_response = response.json()
                except requests.JSONDecodeError as exp:
                    raise ProviderException(
                        response.text, code=response.status_code
                    ) from exp

                generated_text = original_response["choices"][0]["message"]["content"]
                message = [
                    ChatMessageDataClass(role="user", message=text),
                    ChatMessageDataClass(role="system", message=generated_text),
                ]
                standardized_response = ChatDataClass(
                    generated_text=generated_text, message=message
                )

                return ResponseType[ChatDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                )
            else:
                data = response.text
                lst_data = data.split("\n")
                return ResponseType[StreamChat](
                    original_response=None,
                    standardized_response=StreamChat(
                        stream=self.__text_to_json(lst_data)
                    ),
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
        response = self.llm_client.completion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response

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
        response = self.llm_client.acompletion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response
