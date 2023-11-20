from typing import Dict

import requests
from PIL import Image as Img

from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import (
    ExplicitItem,
    ExplicitContentDataClass,
    FaceBoundingBox,
    FaceItem,
    FaceDetectionDataClass,
)
from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceLandmarks,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import standardized_confidence_score_picpurify
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class PicpurifyApi(ProviderInterface, ImageInterface):
    provider_name = "picpurify"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.key = self.api_settings["API_KEY"]
        self.url = "https://www.picpurify.com/analyse/1.1"

    def image__face_detection(
            self, file: str, file_url: str = ""
    ) -> ResponseType[FaceDetectionDataClass]:
        payload = {
            "API_KEY": self.key,
            "task": "face_gender_age_detection",
        }
        file_ = open(file, "rb")
        files = {"image": file_}
        response = requests.post(self.url, files=files, data=payload)
        original_response = response.json()
        file_.close()

        # Handle error
        if "error" in original_response:
            raise ProviderException(
                original_response["error"]["errorMsg"],
                code=response.status_code
            )

        # Std response
        img_size = Img.open(file).size
        width, height = img_size
        face_detection = original_response["face_detection"]["results"]
        faces = []
        for face in face_detection:
            age = face["age_majority"]["decision"]
            if age == "major":
                age = 21.0
            else:
                age = 18.0
            gender = face["gender"]["decision"]
            box = FaceBoundingBox(
                x_min=float(face["face"]["face_rectangle"]["left"] / width),
                x_max=float(face["face"]["face_rectangle"]["right"] / width),
                y_min=float(face["face"]["face_rectangle"]["top"] / height),
                y_max=float(face["face"]["face_rectangle"]["bottom"] / height),
            )
            confidence = face["face"]["confidence_score"]
            faces.append(
                FaceItem(
                    age=age,
                    gender=gender,
                    confidence=confidence,
                    bounding_box=box,
                    # Not supported by picpurify
                    # ---------------------------
                    landmarks=FaceLandmarks(),
                    emotions=FaceEmotions.default(),
                    poses=FacePoses.default(),
                    hair=FaceHair.default(),
                    accessories=FaceAccessories.default(),
                    facial_hair=FaceFacialHair.default(),
                    quality=FaceQuality.default(),
                    makeup=FaceMakeup.default(),
                    occlusions=FaceOcclusions.default(),
                    features=FaceFeatures.default(),
                    # ---------------------------
                )
            )
        standardized_response = FaceDetectionDataClass(items=faces)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def _standardized_confidence(self, confidence_score: float, nsfw: bool) -> float:
        if nsfw:
            return confidence_score
        else:
            return 0.2

    def image__explicit_content(
            self, file: str, file_url: str = ""
    ) -> ResponseType[ExplicitContentDataClass]:
        payload = {
            "API_KEY": self.key,
            "task": "suggestive_nudity_moderation,gore_moderation,"
                    + "weapon_moderation,drug_moderation,hate_sign_moderation",
        }
        file_ = open(file, "rb")
        files = {"image": file_}
        response = requests.post(self.url, files=files, data=payload)
        original_response = response.json()
        file_.close()

        # Handle error
        if "error" in original_response:
            raise ProviderException(
                original_response["error"]["errorMsg"],
                code=response.status_code
            )

        # get moderation label keys from categegories found in image
        # (eg: 'drug_moderation', 'gore_moderation' etc.)
        moderation_labels = original_response.get("performed", [])
        items = []
        for label in moderation_labels:
            classificator = CategoryType.choose_category_subcategory(label.replace("moderation", "content"))
            original_response_label = original_response.get(label, {})
            items.append(
                ExplicitItem(
                    label=label.replace("moderation", "content"),
                    category=classificator["category"],
                    subcategory=classificator["subcategory"],
                    likelihood_score=self._standardized_confidence(
                        original_response_label.get("confidence_score", 0),
                        original_response_label.get(
                            label.replace("moderation", "content"), True)
                    ),
                    likelihood=standardized_confidence_score_picpurify(
                        original_response[label]["confidence_score"],
                        original_response[label][label.replace("moderation", "content")]
                    )
                )
            )

        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        nsfw_score = ExplicitContentDataClass.calculate_nsfw_likelihood_score(items)
        standardized_response = ExplicitContentDataClass(
            items=items, nsfw_likelihood=nsfw, nsfw_likelihood_score=nsfw_score
        )
        res = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return res
