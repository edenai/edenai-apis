from typing import Any, Dict, List, Literal, Type, Union, Optional
import httpx
from openai import BaseModel
from edenai_apis.features.llm.llm_interface import LlmInterface
from edenai_apis.features.llm.chat.chat_dataclass import ChatDataClass
from edenai_apis.features.llm.responses.responses_dataclass import (
    DeleteResponseDataClass,
    ResponsesDataClass,
    StreamResponses,
)
from edenai_apis.features.llm.aresponses.aresponses_dataclass import StreamAResponses


class OpenaiLLMApi(LlmInterface):

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
        response = await self.llm_client.acompletion(
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
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response

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
        response = self.llm_client.responses(
            input=input,
            model=model,
            include=include,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            parallel_tool_calls=parallel_tool_calls,
            prompt=prompt,
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            store=store,
            background=background,
            stream=stream,
            temperature=temperature,
            text=text,
            text_format=text_format,
            tool_choice=tool_choice,
            tools=tools,
            top_p=top_p,
            truncation=truncation,
            user=user,
            service_tier=service_tier,
            safety_identifier=safety_identifier,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            base_url=base_url,
            api_version=api_version,
            custom_llm_provider=custom_llm_provider,
            drop_invalid_params=drop_invalid_params,
            **kwargs,
        )
        return response

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
        response = await self.llm_client.aresponses(
            input=input,
            model=model,
            include=include,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            parallel_tool_calls=parallel_tool_calls,
            prompt=prompt,
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            store=store,
            background=background,
            stream=stream,
            temperature=temperature,
            text=text,
            text_format=text_format,
            tool_choice=tool_choice,
            tools=tools,
            top_p=top_p,
            truncation=truncation,
            user=user,
            service_tier=service_tier,
            safety_identifier=safety_identifier,
            timeout=timeout,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            base_url=base_url,
            api_version=api_version,
            custom_llm_provider=custom_llm_provider,
            drop_invalid_params=drop_invalid_params,
            **kwargs,
        )
        return response

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
        response = self.llm_client.get_responses(
            response_id=response_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        return response

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
        response = await self.llm_client.aget_responses(
            response_id=response_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        return response

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
        response = self.llm_client.delete_responses(
            response_id=response_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        return response

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
        response = await self.llm_client.adelete_responses(
            response_id=response_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        return response
