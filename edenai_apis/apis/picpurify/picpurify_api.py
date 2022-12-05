from io import BufferedReader
import json
from PIL import Image as Img
import requests

from edenai_apis.features import ProviderApi, Image
from edenai_apis.features.image import (
    ExplicitItem,
    ExplicitContentDataClass,
    FaceBoundingBox,
    FaceItem,
    FaceDetectionDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.apis.picpurify.helpers import content_processing


class PicpurifyApi(ProviderApi, Image):

    provider_name = "picpurify"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.key = self.api_settings["API_KEY"]
        self.url = self.api_settings["URL"]

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        payload = {
            "API_KEY": self.key,
            "task": "face_gender_age_detection",
        }
        files = {"image": file}
        response = requests.post(self.url, files=files, data=payload)
        original_response = response.json()

        # Handle error
        if "error" in original_response:
            raise ProviderException(original_response["error"]["errorMsg"])

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
                    age=age, gender=gender, confidence=confidence, bounding_box=box
                )
            )
        standarized_response = FaceDetectionDataClass(items=faces)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        payload = {
            "API_KEY": self.key,
            "task": "suggestive_nudity_moderation,gore_moderation,"
            + "weapon_moderation,drug_moderation,hate_sign_moderation",
        }
        files = {"image": file}
        response = requests.post(self.url, files=files, data=payload)
        original_response = response.json()

        # Handle error
        if "error" in original_response:
            raise ProviderException(original_response["error"]["errorMsg"])

        # get moderation label keys from categegories found in image
        # (eg: 'drug_moderation', 'gore_moderation' etc.)
        moderation_labels = original_response.get("performed", [])

        items = []
        for label in moderation_labels:
            items.append(
                ExplicitItem(
                    label=label.replace("moderation", "content"),
                    likelihood=content_processing(
                        original_response[label]["confidence_score"]
                    ),
                )
            )

        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)

        standarized_response = ExplicitContentDataClass(items=items, nsfw_likelihood=nsfw)
        res = ResponseType[ExplicitContentDataClass](
            original_response=original_response, standarized_response=standarized_response
        )
        print(res.dict())
        return res
