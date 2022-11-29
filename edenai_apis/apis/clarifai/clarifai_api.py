from io import BufferedReader
from typing import Sequence
from google.protobuf.json_format import MessageToDict
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api.status import status_code_pb2
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectItem,
)
from edenai_apis.features.ocr import Bounding_box, OcrDataClass
from edenai_apis.features.image import (
    ExplicitContentDataClass,
    ExplicitItem,
    FaceDetectionDataClass,
    FaceBoundingBox,
    FaceItem,
    ObjectDetectionDataClass,
)
from edenai_apis.features import ProviderApi, Ocr, Image
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

from .clarifai_helpers import explicit_content_likelihood, get_formatted_language


class ClarifaiApi(
    ProviderApi,
    Ocr,
    Image,
):
    provider_name = "clarifai"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.user_id = self.api_settings["user_id"]
        self.app_id = self.api_settings["app_id"]
        self.key = self.api_settings["key"]
        self.explicit_content_code = self.api_settings["explicit_content_code"]
        self.face_detection_code = self.api_settings["face_detection_code"]
        self.object_detection_code = self.api_settings["object_detection_code"]

    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        file_content = file.read()

        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=self.user_id, app_id=self.app_id
        )

        post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            # The user_data_object is created in the overview and is required when using a PAT
            user_app_id=user_data_object,
            model_id=get_formatted_language(language),
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=file_content)
                    )
                )
            ],
        ),
        metadata=metadata,
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise ProviderException("Error calling Clarifai API")


        boxes: Sequence[Bounding_box] = []
        original_response = []

        text = ""
        for output in post_model_outputs_response.outputs:
            original_response.append(str(output.data))
            for region in output.data.regions:
                text += " " + region.data.text.raw
                bb_info = region.region_info.bounding_box
                pos_x1 = float(bb_info.left_col)
                pos_x2 = float(bb_info.right_col)
                pos_y1 = float(bb_info.top_row)
                pos_y2 = float(bb_info.bottom_row)

                boxes.append(
                    Bounding_box(
                        text=region.data.text.raw,
                        left=pos_x1,
                        top=pos_y1,
                        width=pos_x2 - pos_x1,
                        height=pos_y2 - pos_y1,
                    )
                )

        standarized_response = OcrDataClass(
            bounding_boxes=boxes, text=text.replace("\n", " ").strip()
        )
        result = ResponseType[OcrDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        file_content = file.read()

        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=self.user_id, app_id=self.app_id
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                # The user_data_object is created in the overview and is required when using a PAT
                user_app_id=user_data_object,
                model_id=self.explicit_content_code,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=file_content)
                        )
                    )
                ],
            ),
            metadata=metadata,
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise ProviderException("Error calling Clarifai API")
        else:
            response = MessageToDict(
                post_model_outputs_response, preserving_proto_field_name=True
            )
            output = response.get("outputs", None)[0]
            original_response = output["data"]
            items = []
            for concept in output["data"]["concepts"]:
                items.append(
                    ExplicitItem(
                        label=concept["name"],
                        likelihood=explicit_content_likelihood(concept["value"]),
                    )
                )

            result = ResponseType[ExplicitContentDataClass](
                original_response=original_response,
                standarized_response=ExplicitContentDataClass(items=items),
            )
            return result

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        file_content = file.read()

        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=self.user_id, app_id=self.app_id
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                # The user_data_object is created in the overview and is required when using a PAT
                user_app_id=user_data_object,
                model_id=self.face_detection_code,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=file_content)
                        )
                    )
                ],
            ),
            metadata=metadata,
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            print(post_model_outputs_response)
            raise ProviderException(
                "Error calling Clarifai API: "
                + post_model_outputs_response.status.description
            )
        else:
            response = MessageToDict(
                post_model_outputs_response, preserving_proto_field_name=True
            )
            original_reponse = response["outputs"][0]["data"]
            items = []
            for face in original_reponse.get("regions", []):
                rect = face.get("region_info", {}).get("bounding_box")
                item = FaceItem(
                    confidence=face.get("value"),
                    bounding_box=FaceBoundingBox(
                        x_min=rect.get("left_col"),
                        x_max=rect.get("right_col"),
                        y_min=rect.get("top_row"),
                        y_max=rect.get("bottom_row"),
                    ),
                )
                items.append(item)

            result = ResponseType[FaceDetectionDataClass](
                original_response=original_reponse,
                standarized_response=FaceDetectionDataClass(items=items),
            )
            return result

    def image__object_detection(
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        file_content = file.read()

        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=self.user_id, app_id=self.app_id
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                # The user_data_object is created in the overview and is required when using a PAT
                user_app_id=user_data_object,
                model_id=self.object_detection_code,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=file_content)
                        )
                    )
                ],
            ),
            metadata=metadata,
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise ProviderException("Error calling Clarifai API")

        else:
            response = MessageToDict(
                post_model_outputs_response, preserving_proto_field_name=True
            )
            original_response = response["outputs"][0]["data"]

            items = []
            for region in original_response["regions"]:
                rect = region["region_info"]["bounding_box"]
                items.append(
                    ObjectItem(
                        label=region["data"]["concepts"][0]["name"],
                        confidence=region.get("value"),
                        x_min=rect.get("left_col"),
                        x_max=rect.get("right_col"),
                        y_min=rect.get("top_row"),
                        y_max=rect.get("bottom_row"),
                    )
                )

            return ResponseType[ObjectDetectionDataClass](
                original_response=original_response,
                standarized_response=ObjectDetectionDataClass(items=items),
            )
