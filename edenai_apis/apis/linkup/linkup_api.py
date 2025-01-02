from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from linkup import LinkupClient
from typing import Dict, List, Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    type: str
    name: Optional[str]
    url: Optional[str]
    content: Optional[str]


class StandardizedResponse(BaseModel):
    results: List[SearchResult]


class ApiResponse(BaseModel):
    original_response: Dict
    standardized_response: StandardizedResponse


class LinkupApi(ProviderInterface):
    provider_name = "linkup"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.client = LinkupClient(api_key=self.api_settings["api_key"])

    def text__search(
        self, query: str, texts: list = None, similarity_metric: str = "cosine", model: str = None
    ):
        try:
            output_type = "sourcedAnswer"
            depth = "standard"
            
            if model and ":" in model:
                params = model.split(":")
                if len(params) >= 1:
                    output_type = params[0]
                if len(params) >= 2:
                    depth = params[1]

            payload = {
                "query": query,
                "depth": depth,
                "output_type": output_type
            }

            response = self.client.search(**payload)

            if not hasattr(response, "sources") or not isinstance(response.sources, list):
                raise ProviderException("Invalid format for response.sources; expected a list.")

            standardized_response = StandardizedResponse(
                results=[
                    SearchResult(
                        type="text",
                        name=getattr(source, "name", None),
                        url=getattr(source, "url", None),
                        content=getattr(source, "snippet", None),
                    )
                    for source in response.sources
                ]
            )

            return ApiResponse(
                original_response={
                    "answer": response.answer,
                    "sources": response.sources,
                },
                standardized_response=standardized_response
            )

        except Exception as e:
            raise ProviderException(f"Error during Linkup API call: {str(e)}")