from io import BufferedReader
from typing import Dict, Sequence
from google.protobuf.json_format import MessageToDict
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api.status import status_code_pb2
from collections import defaultdict
from PIL import Image as Img
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
    LogoDetectionDataClass,
    LogoItem,
    LogoBoundingPoly,
    LogoVertice
)
from edenai_apis.features import ProviderInterface, OcrInterface, ImageInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    ProviderException,
    LanguageException
)
from edenai_apis.utils.types import ResponseType

from .clarifai_helpers import explicit_content_likelihood, get_formatted_language


class ClarifaiApi(
    ProviderInterface,
    OcrInterface,
    ImageInterface,
):
    provider_name = "clarifai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name, api_keys = api_keys)
        self.user_id = self.api_settings["user_id"]
        self.app_id = self.api_settings["app_id"]
        self.key = self.api_settings["key"]
        self.explicit_content_code = "moderation-recognition"
        self.face_detection_code = "face-detection"

    def ocr__ocr(
        self, 
        file: str, 
        language: str,
        file_url: str= "",
    ) -> ResponseType[OcrDataClass]:

        if not language:
            raise LanguageException("Language not provided")
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        with open(file, "rb") as file_:
            file_content = file_.read()
            
        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id='clarifai', app_id='main'
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

        standardized_response = OcrDataClass(
            bounding_boxes=boxes, text=text.replace("\n", " ").strip()
        )
        result = ResponseType[OcrDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__explicit_content(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ExplicitContentDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        with open(file, "rb") as file_:
            file_content = file_.read()
        user_id = 'clarifai'
        app_id = 'main'
                
        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=user_id, app_id=app_id
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
            raise ProviderException(post_model_outputs_response.status.description)

        response = MessageToDict(
            post_model_outputs_response, preserving_proto_field_name=True
        )
        original_response = response.get("outputs", [])[0]["data"]
        items = []
        for concept in original_response["concepts"]:
            items.append(
                ExplicitItem(
                    label=concept["name"],
                    likelihood=explicit_content_likelihood(concept["value"]),
                )
            )

        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        return ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=ExplicitContentDataClass(items=items, nsfw_likelihood=nsfw),
        )

    def image__face_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        with open(file, "rb") as file_:
            file_content = file_.read()
            
        user_id = 'clarifai'
        app_id = 'main'
        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=user_id, app_id=app_id
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
                standardized_response=FaceDetectionDataClass(items=items),
            )
            return result

    def image__object_detection(
        self, 
        file: str,
        model: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        with open(file, "rb") as file_:
            file_content = file_.read()
            
        user_id = 'clarifai'
        app_id = 'main'
        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=user_id, app_id=app_id
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                # The user_data_object is created in the overview and is required when using a PAT
                user_app_id=user_data_object,
                model_id=model,
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
            default_dict = defaultdict(lambda: None)
            items = []
            regions = original_response.get("regions")
            if regions:
                for region in regions:
                    rect = region["region_info"]["bounding_box"]
                    items.append(
                        ObjectItem(
                            label=region.get("data",default_dict).get(
                                "concepts",[default_dict])[0].get("name"),
                            confidence=region.get("value"),
                            x_min=rect.get("left_col"),
                            x_max=rect.get("right_col"),
                            y_min=rect.get("top_row"),
                            y_max=rect.get("bottom_row"),
                        )
                    )

            return ResponseType[ObjectDetectionDataClass](
                original_response=original_response,
                standardized_response=ObjectDetectionDataClass(items=items),
            )

    def image__logo_detection(
        self, 
        file: str,
        file_url :str = ""
    ) -> ResponseType[LogoDetectionDataClass]:
        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        with open(file, "rb") as file_:
            file_content = file_.read()
        width, height = Img.open(file).size
        user_id = 'clarifai'
        app_id = 'main'
        metadata = (("authorization", self.key),)
        user_data_object = resources_pb2.UserAppIDSet(
            user_id=user_id, app_id=app_id
        )
        model_id = "logos-yolov5"
        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                # The user_data_object is created in the overview and is required when using a PAT
                user_app_id=user_data_object,
                model_id=model_id,
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
            items: Sequence[LogoItem] = []
            regions = original_response.get("regions")
            if regions:
                for region in regions:
                    rect = region["region_info"]["bounding_box"]
                    vertices = []
                    vertices.append(LogoVertice(x=rect.get("left_col",0) * width, y=rect.get("top_row",0) * height))
                    vertices.append(LogoVertice(x=rect.get("right_col",0) * width , y=rect.get("top_row",0) * height))
                    vertices.append(LogoVertice(x=rect.get("right_col",0) * width , y=rect.get("bottom_row",0) * height ))
                    vertices.append(LogoVertice(x=rect.get("left_col",0) * width, y=rect.get("bottom_row",0) * height))
                    items.append(
                        LogoItem(
                            description = region['data']['concepts'][0]['name'],
                            score = region['data']['concepts'][0]['value'],
                            bounding_poly=LogoBoundingPoly(vertices=vertices)
                        )
                    )
            
            return ResponseType[LogoDetectionDataClass](
                original_response=original_response,
                standardized_response=LogoDetectionDataClass(items=items),
            )
