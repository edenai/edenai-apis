import json
import uuid
from typing import Literal, Any, Dict, Optional

import requests

from edenai_apis.apis.google.google_helpers import get_access_token
from edenai_apis.features.multimodal import MultimodalInterface
from edenai_apis.features.multimodal.embeddings import (
    EmbeddingsDataClass,
    EmbeddingModel,
    VideoEmbeddingModel,
)
from edenai_apis.features.multimodal.embeddings.inputsmodel import (
    InputsModel as EmbeddingsInputsModel,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class GoogleMultimodalApi(MultimodalInterface):
    api_settings: Dict[str, Any]
    location: str
    clients: Dict[str, Any]
    project_id: str

    def __construct_header(self) -> Dict[str, str]:
        token = get_access_token(self.location)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def __construct_url(self, model: str, location: str) -> str:
        return (
            "https://"
            + location
            + "-aiplatform.googleapis.com/v1/projects/"
            + self.project_id
            + "/locations/"
            + location
            + "/publishers/google/models/"
            + model
            + ":predict"
        )

    def __upload_file_to_gsc(self, file_path: str) -> str:
        file_extension = file_path.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_extension}"
        storage_client = self.clients["storage"]
        bucket_name = "audios-speech2text"

        # Upload video to GCS
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(filename)

        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{filename}"

    # XXX: Maybe change how dimension is handled, see in constraints and look pricing
    @staticmethod
    def __get_dimension(dimension: str) -> int:
        if dimension == "xs":
            return 128
        if dimension == "s":
            return 256
        if dimension == "m":
            return 512
        if dimension == "xl":
            return 1408
        # TODO: Change error type and message
        raise ValueError("Invalid dimension")

    def __embeddings_construct(
        self, inputs: EmbeddingsInputsModel, dimension: int
    ) -> Dict[str, Any]:
        image_uri = None
        video_uri = None
        if inputs.image:
            image_uri = inputs.image_url

            if not inputs.image_url or not inputs.image_url.startswith("gs://"):
                image_uri = self.__upload_file_to_gsc(inputs.image)

        if inputs.video:
            video_uri = inputs.video_url

            if not inputs.video_url or not inputs.video_url.startswith("gs://"):
                video_uri = self.__upload_file_to_gsc(inputs.video)

        payload: Dict[str, Any] = {
            "instances": [{}],
            "parameters": {"dimension": dimension},
        }

        if inputs.text:
            payload["instances"][0]["text"] = inputs.text

        if image_uri:
            payload["instances"][0]["image"] = {"gcsUri": image_uri}

        if video_uri:
            payload["instances"][0]["video"] = {"gcsUri": video_uri}

        return payload

    def multimodal__embeddings(
        self,
        inputs: Dict[str, Optional[str]],
        model: str,
        dimension: Literal["xs", "s", "m", "xl"] = "xl",
    ) -> ResponseType[EmbeddingsDataClass]:
        location = "us-central1"
        header = self.__construct_header()
        url = self.__construct_url(model, location)

        try:
            inputs_parsed = EmbeddingsInputsModel(**inputs)
        except ValueError as exc:
            raise ProviderException(message="Inputs are not valid") from exc

        payload = self.__embeddings_construct(
            inputs_parsed, GoogleMultimodalApi.__get_dimension(dimension)
        )

        response = requests.post(url, headers=header, json=payload)

        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(message="Parsing Error") from exc

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=EmbeddingsDataClass(
                items=[
                    EmbeddingModel(
                        text_embedding=instance.get("textEmbedding", []),
                        image_embedding=instance.get("imageEmbedding", []),
                        video_embedding=[
                            VideoEmbeddingModel(
                                embedding=video.get("embedding", []),
                                start_offset=video.get("startOffsetSec"),
                                end_offset=video.get("endOffsetSec"),
                            )
                            for video in instance.get("videoEmbeddings", [])
                        ],
                    )
                    for instance in original_response.get("predictions", [])
                ]
            ),
        )
