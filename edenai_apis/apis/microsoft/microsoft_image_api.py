import base64
import json
from typing import List, Sequence, Optional, Any, Dict

import requests
from PIL import Image as Img

from edenai_apis.apis.microsoft.microsoft_helpers import (
    miscrosoft_normalize_face_detection_response,
)
from edenai_apis.apis.microsoft.types import MicrosoftBackgroundRemovalParams
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
from edenai_apis.features.image.background_removal import BackgroundRemovalDataClass
from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.face_recognition.add_face.face_recognition_add_face_dataclass import (
    FaceRecognitionAddFaceDataClass,
)
from edenai_apis.features.image.face_recognition.create_collection.face_recognition_create_collection_dataclass import (
    FaceRecognitionCreateCollectionDataClass,
)
from edenai_apis.features.image.face_recognition.delete_collection.face_recognition_delete_collection_dataclass import (
    FaceRecognitionDeleteCollectionDataClass,
)
from edenai_apis.features.image.face_recognition.delete_face.face_recognition_delete_face_dataclass import (
    FaceRecognitionDeleteFaceDataClass,
)
from edenai_apis.features.image.face_recognition.list_collections.face_recognition_list_collections_dataclass import (
    FaceRecognitionListCollectionsDataClass,
)
from edenai_apis.features.image.face_recognition.list_faces.face_recognition_list_faces_dataclass import (
    FaceRecognitionListFacesDataClass,
)
from edenai_apis.features.image.face_recognition.recognize.face_recognition_recognize_dataclass import (
    FaceRecognitionRecognizeDataClass,
    FaceRecognitionRecognizedFaceDataClass,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class MicrosoftImageApi(ImageInterface):
    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            # Getting response of API
            response = requests.post(
                f"{self.url['vision']}/analyze?visualFeatures=Adult",
                headers=self.headers["vision"],
                data=file_,
            )
            data = response.json()

        # error handling
        if response.status_code != 200:
            if response.status_code == 415:
                # 415 response doesn't have 'error' key
                raise ProviderException(data["message"], code=response.status_code)
            else:
                raise ProviderException(
                    data["error"]["message"], code=response.status_code
                )

        # key is adult but contains all categories (gore, racy, adult)
        moderation_content = data["adult"]

        # Getting the explicit label and its score of image
        items = []
        for explicit_type in ["gore", "adult", "racy"]:
            if moderation_content.get(f"{explicit_type}Score"):
                classificator = CategoryType.choose_category_subcategory(
                    explicit_type.capitalize()
                )
                items.append(
                    ExplicitItem(
                        label=explicit_type.capitalize(),
                        category=classificator["category"],
                        subcategory=classificator["subcategory"],
                        likelihood_score=moderation_content[f"{explicit_type}Score"],
                        likelihood=standardized_confidence_score(
                            moderation_content[f"{explicit_type}Score"]
                        ),
                    )
                )
        nsfw = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        nsfw_score = ExplicitContentDataClass.calculate_nsfw_likelihood_score(items)
        res = ResponseType[ExplicitContentDataClass](
            original_response=data,
            standardized_response=ExplicitContentDataClass(
                items=items, nsfw_likelihood=nsfw, nsfw_likelihood_score=nsfw_score
            ),
        )
        return res

    def image__object_detection(
        self, file: str, model: str = None, file_url: str = "", **kwargs
    ) -> ResponseType[ObjectDetectionDataClass]:
        with open(file, "rb") as file_:
            response = requests.post(
                f"{self.url['vision']}/detect",
                headers=self.headers["vision"],
                data=file_,
            )
            data = response.json()

        if response.status_code != 200:
            error = data["error"]
            err_msg = (
                error["innererror"]["message"]
                if "innererror" in error
                else error["message"]
            )
            raise ProviderException(err_msg, code=response.status_code)

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
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_, Img.open(file) as img:
            file_content = file_.read()
            # Getting size of image
            img_size = img.size

            # Create params for returning face attribute
            params = {
                "recognitionModel": "recognition_04",
                "returnFaceId": "true",
                "returnFaceLandmarks": "true",
                "returnFaceAttributes": (
                    "age,gender,headPose,smile,facialHair,glasses,emotion,"
                    "hair,makeup,occlusion,accessories,blur,exposure,noise"
                ),
            }
            # Getting response of API
            request = requests.post(
                f"{self.url['face']}/detect",
                params=params,
                headers=self.headers["face"],
                data=file_content,
            )
        response = request.json()

        # handle error
        if not isinstance(response, list) and response.get("error") is not None:
            raise ProviderException(
                f'Error calling Microsoft Api: {response["error"].get("message", "error 500")}',
                code=request.status_code,
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
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[LogoDetectionDataClass]:
        with open(file, "rb") as file_:
            response = requests.post(
                f"{self.url['vision']}/analyze?visualFeatures=Brands",
                headers=self.headers["vision"],
                data=file_,
            )
            data = response.json()

        if response.status_code != 200:
            # sometimes no "error" key in repsonse
            # ref: https://westcentralus.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-2/operations/56f91f2e778daf14a499f21b
            error_msg = data.get("message", data.get("error", "message"))
            raise ProviderException(error_msg, code=response.status_code)

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
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[LandmarkDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}analyze?details=Landmarks",
            headers=self.headers["vision"],
            data=file_content,
        ).json()
        items: Sequence[LandmarkItem] = []
        for key in response.get("categories", []):
            for landmark in key.get("detail", {}).get("landmarks", []):
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

    def image__face_recognition__create_collection(
        self, collection_id: str, **kwargs
    ) -> FaceRecognitionCreateCollectionDataClass:
        url = f"{self.url['face']}facelists/{collection_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ],
            "Content-Type": "application/json",
        }
        payload = {"name": collection_id, "recognitionModel": "recognition_04"}
        response = requests.put(url=url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        return FaceRecognitionCreateCollectionDataClass(collection_id=collection_id)

    def image__face_recognition__list_collections(
        self, **kwargs
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        url = f"{self.url['face']}facelists"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ],
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )

        original_response = response.json()
        collections = FaceRecognitionListCollectionsDataClass(
            collections=[face["faceListId"] for face in original_response]
        )
        return ResponseType(
            original_response=original_response, standardized_response=collections
        )

    def image__face_recognition__list_faces(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        original_response = response.json()
        face_ids = [
            face["persistedFaceId"] for face in original_response["persistedFaces"]
        ]
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionListFacesDataClass(face_ids=face_ids),
        )

    def image__face_recognition__delete_collection(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.delete(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        return ResponseType(
            original_response=response.text,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            ),
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}/persistedFaces?detectionModel=detection_03"
        headers = self.headers["face"]
        with open(file, "rb") as file_:
            response = requests.post(url=url, headers=headers, data=file_)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionAddFaceDataClass(
                face_ids=[original_response["persistedFaceId"]]
            ),
        )

    def image__face_recognition__delete_face(
        self, collection_id, face_id, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}/persistedFaces/{face_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.delete(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        return ResponseType(
            original_response=response.text,
            standardized_response=FaceRecognitionDeleteFaceDataClass(deleted=True),
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        # we first need to detect the face, extract the faceId
        # and then make the call for face similarities using this id
        face_detection = self.image__face_detection(file)
        # we get id of the biggest face in the image
        if len(face_detection.standardized_response.items) == 0:
            raise ProviderException("No face detected in the image")
        face_id = face_detection.original_response[0]["faceId"]

        url = f"{self.url['face']}findsimilars"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        payload = {
            "faceId": face_id,
            "faceListId": collection_id,
        }
        response = requests.post(url=url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ProviderException(
                response.json()["error"]["message"], code=response.status_code
            )
        original_response = response.json()
        recognized_faces = [
            FaceRecognitionRecognizedFaceDataClass(
                confidence=face["confidence"], face_id=face["persistedFaceId"]
            )
            for face in original_response
        ]
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionRecognizeDataClass(
                items=recognized_faces
            ),
        )

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        with open(file, "rb") as f:
            if provider_params is None or not isinstance(provider_params, dict):
                microsoft_params = MicrosoftBackgroundRemovalParams()
            else:
                microsoft_params = MicrosoftBackgroundRemovalParams(**provider_params)

            base_url = (
                "https://francecentral.api.cognitive.microsoft.com/computervision/"
            )
            endpoint = "imageanalysis:segment?api-version=2023-02-01-preview"
            url = base_url + endpoint + f"&mode={microsoft_params.mode}"

            response = requests.post(
                url,
                headers=self.headers["vision"],
                data=f.read(),
            )

            if response.status_code != 200:
                try:
                    original_response = response.json()
                    error_message = (
                        original_response["error"]["code"]
                        + ": "
                        + original_response["error"]["message"]
                    )
                except (KeyError, json.JSONDecodeError):
                    error_message = "Provider has not returned an image"

                raise ProviderException(
                    message=error_message,
                    code=response.status_code,
                )

            img_b64 = base64.b64encode(response.content).decode("utf-8")
            resource_url = BackgroundRemovalDataClass.generate_resource_url(img_b64)

            return ResponseType[BackgroundRemovalDataClass](
                original_response=response.text,
                standardized_response=BackgroundRemovalDataClass(
                    image_b64=img_b64,
                    image_resource_url=resource_url,
                ),
            )
