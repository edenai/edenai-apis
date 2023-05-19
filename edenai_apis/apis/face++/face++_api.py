from typing import List
import requests

from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFeatures,
    FaceItem,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
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


# TODO rework error handling (raise message instead of whole response)
# REVIEW change all name instance of Face++ to Facepp (folder name, class name, provider_name etc) ?
class Facepp(ProviderInterface, ImageInterface):
    provider_name = "face++"
    base_url = f"https://api-us.faceplusplus.com/facepp/v3/"

    def __init__(self, api_keys: dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

    def image__face_detection(
        self, file: str, file_url: str = ""
    ) -> ResponseType[FaceDetectionDataClass]:
        if not file_url:
            with open(file, "rb") as f:
                image_payload = {"image_file": f}
        else:
            image_payload = {"image_url": file_url}
        payload = {
            **self.api_settings,
            **image_payload,
            "return_landmark": 2,
            "return_attributes": "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,beauty,mouthstatus,eyegaze,skinstatus",
            "calculate_all": 1,
        }
        response = requests.post(f"{self.base_url}/detect", data=payload)
        if not response.ok:
            raise ProviderException(response.text)

        original_response = response.json()

        face_list: List[FaceItem] = []
        for face in original_response['faces']:
            attributes = face['attribute']
            age = attributes['age']
            gender = attributes['gender']

            # bounding_box
            bb = face['face_rectangle']
            face_bounding_box = FaceBoundingBox(
                x_min=bb['left'],
                x_max=bb['left'] + bb['width'],
                y_max=bb['top'],
                y_min=bb['top'] - bb['height']
            )

            # quality
            face_quality = FaceQuality(
                blur=attributes['blur']['blurness']['value']
            )

            # accessories
            face_accessories = FaceAccessories(
                face_mask=attributes['mouthstatus']['surgical_mask_or_respirator']
            )

            # occlusion
            # TODO eye_occluded=attributes['eyestatus']['occlusion']
            face_occlusion = FaceOcclusions(
                mouth_occluded=attributes['mouthstatus']['other_occlusion'],
            )

            # pose
            pose = attributes['headpose']
            face_pose = FacePoses(
                pitch=pose['pitch_angle'],
                roll=pose['roll_angle'],
                yaw=pose['yaw_angle']
            )

            # face features
            face_features = FaceFeatures(
                smile=attributes['smile']['value']
                mouth_open=attributes['mouthstatus']['open']
                # TODO eyes_open=
            )

            # emotions
            emotions   = attributes['emotion']
            face_emotion = FaceEmotions(
                anger=emotions["anger"],
                disgust=emotions["disgust"],
                fear=emotions["fear"],
                joy=emotions["happiness"],
                neutral=emotions["neutral"],
                sorrow=emotions["sadness"],
                surprise=emotions["surprise"],
            )

            # landmarks
            # contour_left1
            # contour_left2
            # contour_left3
            # contour_left4
            # contour_left5
            # contour_left6
            # contour_left7
            # contour_left8
            # contour_left9
            # contour_left10
            # contour_left11
            # contour_left12
            # contour_left13
            # contour_left14
            # contour_left15
            # contour_left16
            # contour_chin
            # contour_right1
            # contour_right2
            # contour_right3
            # contour_right4
            # contour_right5
            # contour_right6
            # contour_right7
            # contour_right8
            # contour_right9
            # contour_right10
            # contour_right11
            # contour_right12
            # contour_right13
            # contour_right14
            # contour_right15
            # contour_right16

            # left_eyebrow_left_corner
            # left_eyebrow_upper_left_quarter
            # left_eyebrow_upper_middle
            # left_eyebrow_upper_right_quarter
            # left_eyebrow_upper_right_corner
            # left_eyebrow_lower_left_quarter
            # left_eyebrow_lower_middle
            # left_eyebrow_lower_right_quarter
            # left_eyebrow_lower_right_corner

            # right_eyebrow_upper_left_corner
            # right_eyebrow_upper_left_quarter
            # right_eyebrow_upper_middle
            # right_eyebrow_upper_right_quarter
            # right_eyebrow_right_corner
            # right_eyebrow_lower_left_corner
            # right_eyebrow_lower_left_quarter
            # right_eyebrow_lower_middle
            # right_eyebrow_lower_right_quarter

            # nose_bridge1
            # nose_bridge2
            # nose_bridge3
            # nose_tip
            # nose_left_contour1
            # nose_left_contour2
            # nose_left_contour3
            # nose_left_contour4
            # nose_left_contour5
            # nose_middle_contour
            # nose_right_contour1
            # nose_right_contour2
            # nose_right_contour3
            # nose_right_contour4
            # nose_right_contour5

            # left_eye_left_corner
            # left_eye_upper_left_quarter
            # left_eye_top
            # left_eye_upper_right_quarter
            # left_eye_right_corner
            # left_eye_lower_right_quarter
            # left_eye_bottom
            # left_eye_lower_left_quarter
            # left_eye_pupil
            # left_eye_center

            # right_eye_left_corner
            # right_eye_upper_left_quarter
            # right_eye_top
            # right_eye_upper_right_quarter
            # right_eye_right_corner
            # right_eye_lower_right_quarter
            # right_eye_bottom
            # right_eye_lower_left_quarter
            # right_eye_pupil
            # right_eye_center

            # mouth_left_corner
            # mouth_upper_lip_left_contour1
            # mouth_upper_lip_left_contour2
            # mouth_upper_lip_left_contour3
            # mouth_upper_lip_left_contour4
            # mouth_right_corner
            # mouth_upper_lip_right_contour1
            # mouth_upper_lip_right_contour2
            # mouth_upper_lip_right_contour3
            # mouth_upper_lip_right_contour4
            # mouth_upper_lip_top
            # mouth_upper_lip_bottom
            # mouth_lower_lip_right_contour1
            # mouth_lower_lip_right_contour2
            # mouth_lower_lip_right_contour3
            # mouth_lower_lip_left_contour1
            # mouth_lower_lip_left_contour2
            # mouth_lower_lip_left_contour3
            # mouth_lower_lip_top
            # mouth_lower_lip_bottom




    def image__face_recognition__create_collection(
        self, collection_id: str
    ) -> FaceRecognitionCreateCollectionDataClass:
        payload = {**self.api_settings, "outer_id": collection_id}

        response = requests.post(f"{self.base_url}/faceset/create", json=payload)
        if not response.ok:
            raise ProviderException(response.text)

        return FaceRecognitionCreateCollectionDataClass(
            collection_id=response.json()["outer_id"]
        )

    def image__face_recognition__list_collections(
        self,
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        response = requests.post(
            f"{self.base_url}/faceset/getfacesets", json=self.api_settings
        )
        if not response.ok:
            raise ProviderException(response.text)

        original_response = response.json()
        collections = [faceset["outer_id"] for faceset in original_response["facesets"]]
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionListCollectionsDataClass(
                collections=collections
            ),
        )

    def image__face_recognition__delete_collection(
        self, collection_id: str
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        payload = {**self.api_settings, "outer_id": collection_id}

        response = requests.post(f"{self.base_url}/faceset/delete", json=payload)
        if not response.ok:
            raise ProviderException(response.text)

        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            ),
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: str, file_url: str = ""
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        face_detection_response = self.image__face_detection(file, file_url)

        faces_tokens = [
            face["face_token"]
            for face in face_detection_response.original_response["faces"]
        ]
        if len(faces_tokens) > 5:
            raise ProviderException("Cannot add more than 5 faces at once")
        elif len(faces_tokens) == 0:
            raise ProviderException("No face found in this image")

        payload = {
            **self.api_settings,
            "outer_id": collection_id,
            "face_token": ",".join(faces_tokens),
        }

        response = requests.post(f"{self.base_url}/faceset/addface", json=payload)
        if not response.ok:
            raise ProviderException(response.text)

        # TODO handle error if face wasn't added successfully

        original_response = response.json()
        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionAddFaceDataClass(
                face_ids=faces_tokens
            ),
        )

    def image__face_recognition__list_faces(
        self, collection_id: str
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        payload = {**self.api_settings, "outer_id": collection_id}

        response = requests.post(f"{self.base_url}/faceset/getdetail", json=payload)
        if not response.ok:
            raise ProviderException(response.text)

        original_response = response.json()

        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionListFacesDataClass(
                face_ids=original_response["face_tokens"]
            ),
        )

    def image__face_recognition__delete_face(
        self, collection_id, face_id
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        payload = {
            **self.api_settings,
            "outer_id": collection_id,
            "face_tokens": face_id,
        }

        response = requests.post(f"{self.base_url}/faceset/removeface", json=payload)
        if not response.ok:
            raise ProviderException(response.text)

        original_response = response.json()

        return ResponseType(
            original_response=original_response,
            standardized_response=FaceRecognitionDeleteFaceDataClass(deleted=True),
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: str, file_url: str = ""
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        # REVIEW only handle one of `file_url` `file` or both?
        if not file_url:
            with open(file, "rb") as f:
                image_payload = {"image_file": f}
        else:
            image_payload = {"image_url": file_url}

        payload = {
            **self.api_settings,
            **image_payload,
            "outer_id": collection_id,
        }
        response = requests.post(f"{self.base_url}/search", data=payload)
        if not response.ok:
            raise ProviderException(response.text)

        original_result = response.json()
        found_faces = original_result["results"]
        faces = [
            FaceRecognitionRecognizedFaceDataClass(
                confidence=res["confidence"], face_id=res["face_token"]
            )
            for res in found_faces
        ]

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionRecognizeDataClass(items=faces),
        )
