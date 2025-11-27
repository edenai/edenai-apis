from typing import Any, Coroutine, Dict, List, Literal, Optional, Union


from edenai_apis.llmengine.exceptions.llm_engine_exceptions import RerankClientError
from edenai_apis.llmengine.types.response_types import RerankerResponse


class RerankerClient:

    def __init__(
        self,
        provider_name: str = None,
        model_name: Optional[str] = None,
        provider_config: dict = {},
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.model_capabilities = []
        self.call_params = {}

    async def arerank(
        self,
        model: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[
            Literal["cohere", "together_ai", "azure_ai", "infinity", "litellm_proxy"]
        ] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
        **kwargs,
    ) -> Coroutine[Any, Any, RerankerResponse]:
        raise RerankClientError(
            "Not implemented. Please implement this method in a subclass."
        )
