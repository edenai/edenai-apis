"""
Mixin providing default Responses API implementations for providers using LLMEngine.

Any provider class that has `self.llm_client` (an LLMEngine instance) can inherit
from this mixin to get responses/aresponses/get_responses/delete_responses support
without writing any provider-specific code.

Usage:
    class MyProviderLLMApi(LlmResponsesMixin, LlmInterface):
        ...  # only need to implement llm__chat and llm__achat
"""

from typing import Any, Dict, List, Literal, Optional, Type, Union

import httpx
from openai import BaseModel

from edenai_apis.features.llm.aresponses.aresponses_dataclass import StreamAResponses
from edenai_apis.features.llm.responses.responses_dataclass import (
    ResponsesDataClass,
    StreamResponses,
)


class LlmResponsesMixin:
    """Default Responses API implementation that delegates to self.llm_client."""

    def _get_llm_client(self):
        """Get the LLM client. Override in subclasses if needed (e.g. Google uses self.clients["llm_client"])."""
        return self.llm_client

    def llm__responses(
        self,
        input: Union[str, List],
        model: Optional[str] = None,
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
        if base_url is None:
            base_url = getattr(self, "base_url", None)
        return self._get_llm_client().responses(
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

    async def llm__aresponses(
        self,
        input: Union[str, List],
        model: Optional[str] = None,
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
        if base_url is None:
            base_url = getattr(self, "base_url", None)
        return await self._get_llm_client().aresponses(
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

