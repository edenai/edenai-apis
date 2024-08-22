from typing import List, Dict, Optional

import requests

from edenai_apis.features import ProviderInterface, TextInterface

from edenai_apis.features.text.embeddings import EmbeddingsDataClass, EmbeddingDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException


class JinaApi(ProviderInterface, TextInterface):
    provider_name = "jina"

    def __init__(self, api_keys: Dict = {}, **kwargs):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://jina.ai/embeddings/"
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept-Encoding": "identity"}
        )

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model or "jina-embeddings-v2-base-en"
        resp = self.session.post(  # type: ignore
            self.api_url, json={"input": texts, "model": model}
        ).json()
        if "data" not in resp:
            raise ProviderException(resp["detail"])
        embeddings = resp["data"]
        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore
        # Return just the embeddings
        items = [
            EmbeddingDataClass(embedding=result["embedding"])
            for result in sorted_embeddings
        ]
        standardized_response = EmbeddingsDataClass(items=items)
        return ResponseType[EmbeddingsDataClass](
            original_response=resp,
            standardized_response=standardized_response,
        )
