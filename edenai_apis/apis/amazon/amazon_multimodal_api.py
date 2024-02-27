import base64
import json
from typing import Any, Dict, Optional, Literal

from pydantic import ValidationError

from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.features.multimodal import MultimodalInterface
from edenai_apis.features.multimodal.embeddings import (
    EmbeddingsDataClass,
    EmbeddingModel,
)
from edenai_apis.features.multimodal.embeddings.inputsmodel import (
    InputsModel as EmbeddingsInputsModel,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AmazonMultimodalApi(MultimodalInterface):
    api_settings: Dict[str, Any]
    client: Dict[str, Any]
    storage_client: Dict[str, Any]

    # XXX: Maybe change how dimension is handled, see in constraints and look pricing
    def __get_dimension(self, dimension: Literal["xs", "s", "m", "xl"]) -> int:
        if "s" in dimension:
            return 256
        if dimension == "m":
            return 384
        if dimension == "xl":
            return 1024
        # TODO: Change error type and message
        raise ValueError("Invalid dimension")

    def multimodal__embeddings(
        self,
        inputs: Dict[str, Optional[str]],
        model: str,
        dimension: Literal["xs", "s", "m", "xl"] = "xl",
    ) -> ResponseType[EmbeddingsDataClass]:
        try:
            parsed_inputs = EmbeddingsInputsModel(**inputs)
        except ValidationError as exc:
            raise ProviderException(message="Inputs are not valid") from exc

        payload = {
            "inputText": parsed_inputs.text,
            "embeddingConfig": {"outputEmbeddingLength": 256},
        }

        if parsed_inputs.image:
            with open(parsed_inputs.image, "rb") as fstream:
                file_b64 = base64.b64encode(fstream.read()).decode("utf-8")
                payload["inputImage"] = file_b64

        request_params = {
            "body": json.dumps(payload),
            "modelId": f"amazon.{model}",
        }
        response = handle_amazon_call(
            self.clients["bedrock"].invoke_model, **request_params
        )
        original_response = json.loads(response.get("body").read())

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=EmbeddingsDataClass(
                items=[
                    EmbeddingModel(
                        text_embedding=original_response.get("embedding", [])
                        if parsed_inputs.text
                        else [],
                        image_embedding=original_response.get("embedding", [])
                        if parsed_inputs.image
                        else [],
                    )
                ]
            ),
        )
