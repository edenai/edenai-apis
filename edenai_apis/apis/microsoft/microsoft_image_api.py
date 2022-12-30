from io import BufferedReader
from typing import List, Sequence

import requests
from edenai_apis.apis.microsoft.microsoft_helpers import (
    content_processing,
    miscrosoft_normalize_face_detection_response,
)
from edenai_apis.features.image import (
    ExplicitContentDataClass,
    ExplicitItem,
    FaceDetectionDataClass,
    LandmarkDetectionDataClass,
    LandmarkItem,
    LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoItem,
    LogoVertice,
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from PIL import Image as Img


class MicrosoftImageApi(ImageInterface):
    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionExplicitDetection Object that contains the
        the objects and their location
        """

        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}/analyze?visualFeatures=Adult",
            headers=self.headers["vision"],
            data=file,
        )
        data = response.json()

        # error handling
        if response.status_code != 200:
            if response.status_code == 415:
                # 415 response doesn't have 'error' key
                raise ProviderException(data["message"])
            else:
                raise ProviderException(data["error"]["message"])

        # key is adult but contains all categories (gore, racy, adult)
        moderation_content = data["adult"]

        # Getting the explicit label and its score of image
        items = []
        for explicit_type in ["gore", "adult", "racy"]:
            if moderation_content.get(f"{explicit_type}Score"):
                items.append(
                    ExplicitItem(
                        label=explicit_type.capitalize(),
                        likelihood=content_processing(
                            moderation_content[f"{explicit_type}Score"]
                        ),
                    )
                )
        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)

        res = ResponseType[ExplicitContentDataClass](
            original_response=data,
            standardized_response=ExplicitContentDataClass(
                items=items, nsfw_likelihood=nsfw
            ),
        )
        return res

    def image__object_detection(
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionObjectDetection Object that contains the
        the objects and their location
        """
        # Call api
        response = requests.post(
            f"{self.url['vision']}/detect",
            headers=self.headers["vision"],
            data=file,
        )
        data = response.json()

        if response.status_code != 200:
            error = data["error"]
            err_msg = (
                error["innererror"]["message"] if "innererror" in error else error["message"]
            )
            raise ProviderException(err_msg)

        items = []

        metadata = data.get("metadata", {})
        width, height = metadata.get("width"), metadata.get("height")

        for obj in data.get("objects", []):
            if width is None or height is None:
                x_min, x_max, y_min, y_max = 0, 0, 0, 0
            else:
                x_min = obj["rectangle"]["x"] / width
                x_max = (obj["rectangle"]["x"] + obj["rectangle"]["w"]) / width
                y_min = 1 - ((height - obj["rectangle"]["y"]) / height)
                y_max = 1 - (
                    (height - obj["rectangle"]["y"] - obj["rectangle"]["h"]) / height
                )
            items.append(
                ObjectItem(
                    label=obj["object"],
                    confidence=obj["confidence"],
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        return ResponseType[ObjectDetectionDataClass](
            original_response=data,
            standardized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        :return:            VisionFaceDetection Object that contains the
        the objects and their location
        """

        file_content = file.read()
        # Getting size of image
        img_size = Img.open(file).size

        # Create params for returning face attribute
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "true",
            "returnFaceAttributes": (
                "age,gender,headPose,smile,facialHair,glasses,emotion,"
                "hair,makeup,occlusion,accessories,blur,exposure,noise"
            ),
        }
        # Getting response of API
        response = requests.post(
            f"{self.url['face']}/detect",
            params=params,
            headers=self.headers["face"],
            data=file_content,
        ).json()

        # handle error
        if not isinstance(response, list) and response.get("error") is not None:
            print(response)
            raise ProviderException(
                f'Error calling Microsoft Api: {response["error"].get("message", "error 500")}'
            )
        # Create response VisionFaceDetection object

        faces_list: List = miscrosoft_normalize_face_detection_response(
            response, img_size
        )

        return ResponseType[FaceDetectionDataClass](
            original_response=response,
            standardized_response=FaceDetectionDataClass(items=faces_list),
        )

    def image__logo_detection(
        self, file: BufferedReader
    ) -> ResponseType[LogoDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        """
        response = requests.post(
            f"{self.url['vision']}/analyze?visualFeatures=Brands",
            headers=self.headers["vision"],
            data=file,
        )
        data = response.json()

        if response.status_code != 200:
            # sometimes no "error" key in repsonse
            # ref: https://westcentralus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/56f91f2e778daf14a499f21b
            error_msg = data.get("message", data.get("error", "message"))
            raise ProviderException(error_msg)

        items: Sequence[LogoItem] = []
        for key in data.get("brands"):
            x_cordinate = float(key.get("rectangle").get("x"))
            y_cordinate = float(key.get("rectangle").get("y"))
            height = float(key.get("rectangle").get("h"))
            weidth = float(key.get("rectangle").get("w"))
            vertices = []
            vertices.append(LogoVertice(x=x_cordinate, y=y_cordinate))
            vertices.append(LogoVertice(x=x_cordinate + weidth, y=y_cordinate))
            vertices.append(LogoVertice(x=x_cordinate + weidth, y=y_cordinate + height))
            vertices.append(LogoVertice(x=x_cordinate, y=y_cordinate + height))

            items.append(
                LogoItem(
                    description=key.get("name"),
                    score=key.get("confidence"),
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )

        return ResponseType[LogoDetectionDataClass](
            original_response=data,
            standardized_response=LogoDetectionDataClass(items=items),
        )

    def image__landmark_detection(
        self, file: BufferedReader
    ) -> ResponseType[LandmarkDetectionDataClass]:
        """
        :param image_path:  String that contains the path to the image file
        """

        file_content = file.read()

        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}analyze?details=Landmarks",
            headers=self.headers["vision"],
            data=file_content,
        ).json()
        items: Sequence[LandmarkItem] = []
        for key in response.get("categories"):
            for landmark in key.get("detail", {}).get("landmarks"):
                if landmark.get("name") not in [item.description for item in items]:
                    items.append(
                        LandmarkItem(
                            description=landmark.get("name"),
                            confidence=landmark.get("confidence"),
                        )
                    )

        return ResponseType[LandmarkDetectionDataClass](
            original_response=response,
            standardized_response=LandmarkDetectionDataClass(items=items),
        )
