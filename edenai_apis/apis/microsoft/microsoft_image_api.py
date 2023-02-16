from io import BufferedReader
from typing import List, Sequence

import requests
from edenai_apis.apis.microsoft.microsoft_helpers import (
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
from edenai_apis.features.image.face_recognition.add_face.face_recognition_add_face_dataclass import (
    FaceRecognitionAddFaceDataClass,
)
from edenai_apis.features.image.face_recognition.create_collection.face_recognition_create_collection_dataclass import (
    FaceRecognitionCreateCollectionDataClass,
)
from edenai_apis.features.image.face_recognition.delete_collection.face_recognition_delete_collection_dataclass import FaceRecognitionDeleteCollectionDataClass
from edenai_apis.features.image.face_recognition.delete_face.face_recognition_delete_face_dataclass import FaceRecognitionDeleteFaceDataClass
from edenai_apis.features.image.face_recognition.recognize.face_recognition_recognize_dataclass import (
    FaceRecognitionRecognizeDataClass,
    FaceRecognitionRecognizedFaceDataClass,
)
from edenai_apis.features.image.face_recognition.list_collections.face_recognition_list_collections_dataclass import (
    FaceRecognitionListCollectionsDataClass,
)
from edenai_apis.features.image.face_recognition.list_faces.face_recognition_list_faces_dataclass import (
    FaceRecognitionListFacesDataClass,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from PIL import Image as Img


class MicrosoftImageApi(ImageInterface):
    def image__explicit_content(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ExplicitContentDataClass]:

        file_ = open(file, "rb")
        # Getting response of API
        response = requests.post(
            f"{self.url['vision']}/analyze?visualFeatures=Adult",
            headers=self.headers["vision"],
            data=file_,
        )
        data = response.json()
        file_.close()

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
                        likelihood=standardized_confidence_score(
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
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:

        file_ = open(file, "rb")
        response = requests.post(
            f"{self.url['vision']}/detect",
            headers=self.headers["vision"],
            data=file_,
        )
        data = response.json()
        file_.close()

        if response.status_code != 200:
            error = data["error"]
            err_msg = (
                error["innererror"]["message"]
                if "innererror" in error
                else error["message"]
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
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:

        file_ = open(file, "rb")
        file_content = file_.read()
        # Getting size of image
        img_size = Img.open(file).size

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

        file_content = file.read()

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
        self, collection_id: str
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
            raise ProviderException(response.json()["error"]["message"])
        return FaceRecognitionCreateCollectionDataClass(collection_id=collection_id)

    def image__face_recognition__list_collections(
        self,
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        url = f"{self.url['face']}facelists"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ],
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(response.json()["error"]["message"])

        original_response = response.json()
        collections = FaceRecognitionListCollectionsDataClass(
            collections=[face["faceListId"] for face in original_response]
        )
        return ResponseType(
            original_response=original_response, standardized_response=collections
        )

    def image__face_recognition__list_faces(
        self, collection_id: str
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.get(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(response.json()["error"]["message"])
        original_response = response.json()
        face_ids = [
            face["persistedFaceId"] for face in original_response["persistedFaces"]
        ]
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionListFacesDataClass(
                face_ids=face_ids
            ),
        )

    def image__face_recognition__delete_collection(
            self, collection_id: str
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.delete(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(response.json()["error"]["message"])
        return ResponseType(
            original_response=response.text,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            )
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: BufferedReader
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}/persistedFaces?detectionModel=detection_03"
        headers = self.headers["face"]
        response = requests.post(url=url, headers=headers, data=file)
        if response.status_code != 200:
            raise ProviderException(response.json()["error"]["message"])
        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionAddFaceDataClass(
                face_ids=[original_response["persistedFaceId"]]
            ),
        )

    def image__face_recognition__delete_face(
            self, collection_id, face_id
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        url = f"{self.url['face']}facelists/{collection_id}/persistedFaces/{face_id}"
        headers = {
            "Ocp-Apim-Subscription-Key": self.headers["face"][
                "Ocp-Apim-Subscription-Key"
            ]
        }
        response = requests.delete(url=url, headers=headers)
        if response.status_code != 200:
            raise ProviderException(response.json()["error"]["message"])
        return ResponseType(
            original_response=response.text,
            standardized_response=FaceRecognitionDeleteFaceDataClass(
                deleted=True
            )
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: BufferedReader
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
            raise ProviderException(response.json()["error"]["message"])
        original_response = response.json()
        recognized_faces = [
            FaceRecognitionRecognizedFaceDataClass(
                confidence=face["confidence"], face_id=face["persistedFaceId"]
            )
            for face in original_response
        ]
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionRecognizeDataClass(items=recognized_faces),
        )
