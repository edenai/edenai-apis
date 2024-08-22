from typing import Dict, List

import requests

from edenai_apis.features import ProviderInterface
from edenai_apis.features.image import FaceItem, FaceDetectionDataClass
from edenai_apis.features.image import ImageInterface
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceLandmarks,
    FaceBoundingBox,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class SkybiometryApi(ProviderInterface, ImageInterface):
    provider_name = "skybiometry"

    def __init__(self, api_keys: Dict = {}, **kwargs) -> None:
        self.base_url = "https://api.skybiometry.com/fc/"
        self.settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.settings["api_key"]
        self.api_secret = self.settings["api_secret"]

    def image__face_detection(
        self, file: str, file_url: str = ""
    ) -> ResponseType[FaceDetectionDataClass]:
        file_ = open(file, "rb")
        files = {"file": file_}

        endpoint = f"{self.base_url}faces/detect.json"
        query_params = (
            f"api_key={self.api_key}&api_secret={self.api_secret}&attributes=all"
        )
        response = requests.post(f"{endpoint}?{query_params}", files=files)

        original_response = response.json()
        file_.close()
        if response.status_code != 200:
            raise ProviderException(
                message=original_response["error_message"], code=response.status_code
            )

        if len(original_response["photos"][0]["tags"]) > 0:
            original_response = original_response["photos"][0]["tags"]
        else:
            raise ProviderException(
                message="Provider did not return any face", code=404
            )

        items: List[FaceItem] = []
        for face in original_response:
            items.append(
                FaceItem(
                    confidence=face["attributes"].get("face", {}).get("confidence"),
                    landmarks=FaceLandmarks(
                        left_eye=[
                            face.get("eye_left", {}).get("x", 0),
                            face.get("eye_left", {}).get("y", 0),
                        ],
                        right_eye=[
                            face.get("eye_right", {}).get("x", 0),
                            face.get("eye_right", {}).get("y", 0),
                        ],
                        mouth_center=[
                            face.get("mouse_center", {}).get("x", 0),
                            face.get("mouse_center", {}).get("y", 0),
                        ],
                        nose_tip=[
                            face.get("eye_left", {}).get("x", 0),
                            face.get("eye_left", {}).get("y", 0),
                        ],
                    ),
                    bounding_box=FaceBoundingBox(
                        x_min=face["center"]["x"] - face["width"] / 2,
                        x_max=face["center"]["x"] + face["width"] / 2,
                        y_min=face["center"]["y"] - face["height"] / 2,
                        y_max=face["center"]["y"] + face["height"] / 2,
                    ),
                    emotions=FaceEmotions(
                        joy=face["attributes"].get("happiness", {}).get("confidence"),
                        sorrow=face["attributes"].get("sadness", {}).get("confidence"),
                        anger=face["attributes"].get("anger", {}).get("confidence"),
                        surprise=face["attributes"]
                        .get("surprise", {})
                        .get("confidence"),
                        disgust=face["attributes"].get("disgust", {}).get("confidence"),
                        fear=face["attributes"].get("fear", {}).get("confidence"),
                        confusion=None,
                        calm=None,
                        unknown=None,
                        neutral=face["attributes"]
                        .get("neutral_mood", {})
                        .get("confidence"),
                        contempt=None,
                    ),
                    age=face["attributes"].get("age_est", {}).get("value"),
                    gender=face["attributes"].get("gender", {}).get("value"),
                    facial_hair=FaceFacialHair(
                        moustache=(
                            1.0
                            if face["attributes"].get("mustache", {}).get("value")
                            == "true"
                            else 0.0
                        ),
                        beard=(
                            1.0
                            if face["attributes"].get("beard", {}).get("value")
                            == "true"
                            else 0.0
                        ),
                        sideburns=None,
                    ),
                    accessories=FaceAccessories(
                        sunglasses=(
                            1.0
                            if face["attributes"].get("dark_glasses", {}).get("value")
                            == "true"
                            else 0.0
                        ),
                        eyeglasses=(
                            1.0
                            if face["attributes"].get("glasses", {}).get("value")
                            == "true"
                            else 0.0
                        ),
                        headwear=(
                            1.0
                            if face["attributes"].get("hat", {}).get("value") == "true"
                            else 0.0
                        ),
                        reading_glasses=None,
                        swimming_goggles=None,
                        face_mask=None,
                    ),
                    features=FaceFeatures(
                        eyes_open=(
                            1.0
                            if face["attributes"].get("eyes", {}).get("value") == "open"
                            else 0.0
                        ),
                        smile=(
                            1.0
                            if face["attributes"].get("smiling", {}).get("value")
                            == "true"
                            else 0.0
                        ),
                        mouth_open=(
                            1.0
                            if face["attributes"].get("lips", {}).get("value", "sealed")
                            != "sealed"
                            else 0.0
                        ),
                    ),
                    poses=FacePoses.default(),
                    quality=FaceQuality.default(),
                    occlusions=FaceOcclusions.default(),
                    hair=FaceHair.default(),
                    makeup=FaceMakeup.default(),
                )
            )

        standardized_response = FaceDetectionDataClass(items=items)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
