from typing import List, Optional

import requests

from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.face_compare.face_compare_dataclass import (
    FaceCompareDataClass,
    FaceCompareBoundingBox,
    FaceMatch,
)
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
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class FaceppApi(ProviderInterface, ImageInterface):
    provider_name = "facepp"
    base_url = f"https://api-us.faceplusplus.com/facepp/v3"

    def __init__(self, api_keys: dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

    def _get_face_tokens(self, file: str, file_url: Optional[str] = None) -> List[str]:
        if file_url:
            response = requests.post(
                f"{self.base_url}/detect",
                data={**self.api_settings, "image_url": file_url},
            )
        else:
            with open(file, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/detect",
                    data=self.api_settings,
                    files={"image_file": f},
                )
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        return [face["face_token"] for face in original_response["faces"]]

    def image__face_recognition__create_collection(
        self, collection_id: str, **kwargs
    ) -> FaceRecognitionCreateCollectionDataClass:
        payload = {**self.api_settings, "outer_id": collection_id}
        response = requests.post(f"{self.base_url}/faceset/create", data=payload)
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        return FaceRecognitionCreateCollectionDataClass(
            collection_id=response.json()["outer_id"]
        )

    def image__face_recognition__list_collections(
        self, **kwargs
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        response = requests.post(
            f"{self.base_url}/faceset/getfacesets", data=self.api_settings
        )
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        collections = [faceset["outer_id"] for faceset in original_response["facesets"]]
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionListCollectionsDataClass(
                collections=collections
            ),
        )

    def image__face_recognition__delete_collection(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        payload = {**self.api_settings, "outer_id": collection_id, "check_empty": 0}

        response = requests.post(f"{self.base_url}/faceset/delete", data=payload)
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            ),
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: str, file_url: Optional[str] = None, **kwargs
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        faces_tokens = self._get_face_tokens(file, file_url)

        if len(faces_tokens) > 5:
            raise ProviderException("Cannot add more than 5 faces at once")
        elif len(faces_tokens) == 0:
            raise ProviderException("No face found in this image")

        payload = {
            **self.api_settings,
            "outer_id": collection_id,
            "face_tokens": ",".join(faces_tokens),
        }

        response = requests.post(f"{self.base_url}/faceset/addface", data=payload)
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionAddFaceDataClass(
                face_ids=faces_tokens
            ),
        )

    def image__face_recognition__list_faces(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        payload = {**self.api_settings, "outer_id": collection_id}

        response = requests.post(f"{self.base_url}/faceset/getdetail", data=payload)
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()

        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionListFacesDataClass(
                face_ids=original_response["face_tokens"]
            ),
        )

    def image__face_recognition__delete_face(
        self, collection_id, face_id, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        payload = {
            **self.api_settings,
            "outer_id": collection_id,
            "face_tokens": face_id,
        }

        response = requests.post(f"{self.base_url}/faceset/removeface", data=payload)
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()

        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionDeleteFaceDataClass(deleted=True),
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: str, file_url: Optional[str] = None, **kwargs
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        payload = {
            **self.api_settings,
            "outer_id": collection_id,
        }
        if file_url:
            response = requests.post(
                f"{self.base_url}/search", data={"image_url": file_url, **payload}
            )
        else:
            with open(file, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/search",
                    data=payload,
                    files={"image_file": f},
                )
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        found_faces = original_response.get("results", [])
        faces = [
            FaceRecognitionRecognizedFaceDataClass(
                confidence=res["confidence"], face_id=res["face_token"]
            )
            for res in found_faces
        ]

        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionRecognizeDataClass(items=faces),
        )

    def image__face_compare(
        self,
        file1: str,
        file2: str,
        file1_url: Optional[str] = None,
        file2_url: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[FaceCompareDataClass]:
        url = self.base_url + "/compare"
        if file1_url and file2_url:
            payload = {
                **self.api_settings,
                "image_url1": file1_url,
                "image_url2": file2_url,
            }
            response = requests.post(url, data=payload)
        else:
            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                response = requests.post(
                    url=url,
                    data=self.api_settings,
                    files={
                        "image_file1": f1,
                        "image_file2": f2,
                    },
                )
        if not response.ok:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        faces = []
        for matching_face in original_response.get("faces2"):
            confidence = original_response.get("confidence") or 0
            faces.append(
                FaceMatch(
                    confidence=confidence / 100,
                    bounding_box=FaceCompareBoundingBox(
                        top=matching_face.get("face_rectangle").get("top"),
                        left=matching_face.get("face_rectangle").get("left"),
                        height=matching_face.get("face_rectangle").get("height"),
                        width=matching_face.get("face_rectangle").get("width"),
                    ),
                )
            )
        standardized_response = FaceCompareDataClass(items=faces)

        return ResponseType[FaceCompareDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
