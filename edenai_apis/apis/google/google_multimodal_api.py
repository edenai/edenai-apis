from typing import Literal, Any, Dict

from edenai_apis.apis.google.google_helpers import get_access_token
from edenai_apis.features.multimodal import MultimodalInterface
from edenai_apis.features.multimodal.embeddings import EmbeddingsDataClass
from edenai_apis.features.multimodal.embeddings.inputsmodel import (
    InputsModel as EmbeddingsInputsModel,
)
from edenai_apis.utils.types import ResponseType


class GoogleMultimodalApi(MultimodalInterface):
    api_settings: Dict[str, Any]
    location: str
    clients: Dict[str, Any]
    project_id: str

    def multimodal__embeddings(
        self,
        inputs: EmbeddingsInputsModel,
        dimension: Literal["xs", "s", "m", "xl"] = "xl",
    ) -> ResponseType[EmbeddingsDataClass]:
        token = get_access_token(self.location)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        url = f"https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/publishers/google/models/multimodalembedding@001:predict"
