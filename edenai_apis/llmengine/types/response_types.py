from typing import List, Optional, Union
from litellm import CustomStreamWrapper
from litellm.types.utils import ModelResponse, EmbeddingResponse, Embedding, Usage


class EmbeddingResponseModel(EmbeddingResponse):
    """
    Example response for an embedding request
    {
        "data": [
            {
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }
    """

    provider_time: int = 0
    edenai_time: int = 0

    def __init__(
        self,
        model: Optional[str] = None,
        usage: Optional[Usage] = None,
        response_ms=None,
        data: Optional[Union[List, List[Embedding]]] = None,
        hidden_params=None,
        _response_headers=None,
        provider_time=None,
        edenai_time=None,
        **params,
    ):
        super().__init__(
            data=data,
            model=model,
            object=object,
            usage=usage,
            response_ms=response_ms,
            hidden_params=hidden_params,
            **params,
        )


class ResponseModel(ModelResponse):
    """
    Example response for a simple chat completion
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!"
                },
                "logprobs": null,
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
    """

    provider_time: int = 0
    edenai_time: int = 0

    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        object=None,
        system_fingerprint=None,
        usage=None,
        stream=None,
        stream_options=None,
        response_ms=None,
        hidden_params=None,
        provider_time=None,
        edenai_time=None,
        **params,
    ):
        super().__init__(
            id=id,
            choices=choices,
            created=created,
            model=model,
            object=object,
            system_fingerprint=system_fingerprint,
            usage=usage,
            stream=stream,
            stream_options=stream_options,
            response_ms=response_ms,
            hidden_params=hidden_params,
            **params,
        )

        self.provider_time = provider_time
        self.edenai_time = edenai_time


class CustomStreamWrapperModel(CustomStreamWrapper):
    def __init__(
        self,
        rcompletion_stream,
        model,
        custom_llm_provider=None,
        logging_obj=None,
        stream_options=None,
    ):
        super().__init__(
            rcompletion_stream,
            model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            stream_options=stream_options,
        )
