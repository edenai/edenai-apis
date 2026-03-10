"""
Global helper functions for retrieving and deleting stored LLM responses.

These bypass the Feature/subfeature/provider pipeline entirely — they
instantiate an LLMEngine directly and call litellm's get/delete responses.
No pricing rows or subfeature registration required.

Usage (async)::

    from edenai_apis.features.llm.response_helpers import aget_response, adelete_response

    result = await aget_response("resp_abc123", "openai", {"api_key": "sk-..."})
    deleted = await adelete_response("resp_abc123", "openai", {"api_key": "sk-..."})
"""

from typing import Any, Dict, Optional, Union

import httpx

from edenai_apis.features.llm.responses.responses_dataclass import (
    DeleteResponseDataClass,
    ResponsesDataClass,
)
from edenai_apis.llmengine.llm_engine import LLMEngine


def get_response(
    response_id: str,
    provider_name: str,
    provider_config: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> ResponsesDataClass:
    engine = LLMEngine(
        provider_name=provider_name,
        provider_config=provider_config or {},
    )
    return engine.get_responses(
        response_id=response_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )


async def aget_response(
    response_id: str,
    provider_name: str,
    provider_config: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> ResponsesDataClass:
    engine = LLMEngine(
        provider_name=provider_name,
        provider_config=provider_config or {},
    )
    return await engine.aget_responses(
        response_id=response_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )


def delete_response(
    response_id: str,
    provider_name: str,
    provider_config: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> DeleteResponseDataClass:
    engine = LLMEngine(
        provider_name=provider_name,
        provider_config=provider_config or {},
    )
    return engine.delete_responses(
        response_id=response_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )


async def adelete_response(
    response_id: str,
    provider_name: str,
    provider_config: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> DeleteResponseDataClass:
    engine = LLMEngine(
        provider_name=provider_name,
        provider_config=provider_config or {},
    )
    return await engine.adelete_responses(
        response_id=response_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
    )
