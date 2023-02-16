from io import BufferedReader
from typing import List

from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceItem,
    FaceLandmarks,
    FacePoses,
    FaceQuality,
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
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AmazonImageApi(ImageInterface):
    def image__object_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:

        with open(file, "rb") as file_:
            file_content = file_.read()
        # Getting API response
        try:
            original_response = self.clients["image"].detect_labels(
                Image={"Bytes": file_content}, MinConfidence=70
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        # Standarization
        items = []
        for object_label in original_response.get("Labels"):

            if object_label.get("Instances"):
                bounding_box = object_label.get("Instances")[0].get("BoundingBox")
                x_min, x_max = (
                    bounding_box.get("Left"),
                    bounding_box.get("Left") + bounding_box.get("Width"),
                )
                y_min, y_max = (
                    bounding_box.get("Top"),
                    bounding_box.get("Top") + bounding_box.get("Height"),
                )
            else:
                x_min, x_max, y_min, y_max = None, None, None, None

            items.append(
                ObjectItem(
                    label=object_label.get("Name"),
                    confidence=object_label.get("Confidence") / 100,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        return ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standardized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:
        
        with open(file, "rb") as file_:
            file_content = file_.read()

        # Getting Response
        original_response = self.clients["image"].detect_faces(
            Image={"Bytes": file_content}, Attributes=["ALL"]
        )

        # Standarize Response
        faces_list = []
        for face in original_response.get("FaceDetails", []):

            # Age
            age_output = None
            age_range = face.get("AgeRange")
            if age_range:
                age_output = (
                    age_range.get("Low", 0.0) + age_range.get("High", 100)
                ) / 2

            # features
            features = FaceFeatures(
                eyes_open=face.get("eyes_open", {}).get("Confidence", 0.0) / 100,
                smile=face.get("smile", {}).get("Confidence", 0.0) / 100,
                mouth_open=face.get("mouth_open", {}).get("Confidence", 0.0) / 100,
            )

            # accessories
            accessories = FaceAccessories(
                sunglasses=face.get("Sunglasses", {}).get("Confidence", 0.0) / 100,
                eyeglasses=face.get("Eyeglasses", {}).get("Confidence", 0.0) / 100,
            )

            # facial hair
            facial_hair = FaceFacialHair(
                moustache=face.get("Mustache", {}).get("Confidence", 0.0) / 100,
                beard=face.get("Beard", {}).get("Confidence", 0.0) / 100,
            )

            # quality
            quality = FaceQuality(
                brightness=face.get("Quality").get("Brightness", 0.0) / 100,
                sharpness=face.get("Quality").get("Sharpness", 0.0) / 100,
            )

            # emotions
            emotion_output = {}
            for emo in face.get("Emotions", []):
                normalized_emo = emo.get("Confidence", 0.0) * 100
                if emo.get("Type"):
                    if emo.get("Type").lower() == "happy":  # normalise keywords
                        emo["Type"] = "happiness"
                    emotion_output[emo.get("Type").lower()] = standardized_confidence_score(
                        normalized_emo / 100
                    )
            emotions = FaceEmotions(
                anger=emotion_output.get("angry"),
                surprise=emotion_output.get("surprise"),
                fear=emotion_output.get("fear"),
                sorrow=emotion_output.get("sadness"),
                confusion=emotion_output.get("confused"),
                calm=emotion_output.get("calm"),
                disgust=emotion_output.get("disgsusted"),
                joy=emotion_output.get("happiness"),
            )

            # landmarks
            landmarks_output = {}
            for land in face.get("Landmarks"):
                if land.get("Type") and land.get("X") and land.get("Y"):
                    landmarks_output[land.get("Type")] = [land.get("X"), land.get("Y")]

            landmarks = FaceLandmarks(
                left_eye=landmarks_output.get("eye_left", []),
                left_eye_top=landmarks_output.get("eye_leftUp", []),
                left_eye_right=landmarks_output.get("lefteye_right", []),
                left_eye_bottom=landmarks_output.get("leftEyeDown", []),
                left_eye_left=landmarks_output.get("leftEyeLeft", []),
                right_eye=landmarks_output.get("eye_right", []),
                right_eye_top=landmarks_output.get("eye_rightUp", []),
                right_eye_right=landmarks_output.get("eye_rightRight", []),
                right_eye_bottom=landmarks_output.get("rightEyeDown", []),
                right_eye_left=landmarks_output.get("rightEyeLeft", []),
                left_eyebrow_left=landmarks_output.get("leftEyeBrowLeft", []),
                left_eyebrow_right=landmarks_output.get("leftEyeBrowRight", []),
                left_eyebrow_top=landmarks_output.get("leftEyeBrowUp", []),
                right_eyebrow_left=landmarks_output.get("rightEyeBrowLeft", []),
                right_eyebrow_right=landmarks_output.get("rightEyeBrowRight", []),
                right_eyebrow_top=landmarks_output.get("rightEyeBrowUp", []),
                left_pupil=landmarks_output.get("leftPupil", []),
                right_pupil=landmarks_output.get("rightPupil", []),
                nose_tip=landmarks_output.get("nose", []),
                nose_bottom_right=landmarks_output.get("noseRight", []),
                nose_bottom_left=landmarks_output.get("noseLeft", []),
                mouth_left=landmarks_output.get("mouth_left", []),
                mouth_right=landmarks_output.get("mouth_right", []),
                mouth_top=landmarks_output.get("mouthUp", []),
                mouth_bottom=landmarks_output.get("mouthDown", []),
                chin_gnathion=landmarks_output.get("chinBottom", []),
                upper_jawline_left=landmarks_output.get("upperJawlineLeft", []),
                mid_jawline_left=landmarks_output.get("midJawlineLeft", []),
                mid_jawline_right=landmarks_output.get("midJawlineRight", []),
                upper_jawline_right=landmarks_output.get("upperJawlineRight", []),
            )
            poses = FacePoses(
                roll=face.get("Pose", {}).get("Roll"),
                yaw=face.get("Pose", {}).get("Yaw"),
                pitch=face.get("Pose", {}).get("Pitch"),
            )

            faces_list.append(
                FaceItem(
                    age=age_output,
                    gender=face.get("Gender", {}).get("Value"),
                    facial_hair=facial_hair,
                    features=features,
                    accessories=accessories,
                    quality=quality,
                    emotions=emotions,
                    landmarks=landmarks,
                    poses=poses,
                    confidence=face.get("Confidence", 0.0) / 100,
                    bounding_box=FaceBoundingBox(
                        x_min=face.get("BoundingBox", {}).get("Left", 0.0),
                        x_max=face.get("BoundingBox", {}).get("Left", 0.0)
                        + face.get("BoundingBox", {}).get("Width", 0.0),
                        y_min=face.get("BoundingBox", {}).get("Top", 0.0),
                        y_max=face.get("BoundingBox", {}).get("Top", 0.0)
                        + face.get("BoundingBox", {}).get("Height", 0.0),
                    ),
                )
            )

        standardized_response = FaceDetectionDataClass(items=faces_list)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        file_content = file.read()

        try:
            response = self.clients["image"].detect_moderation_labels(
                Image={"Bytes": file_content}, MinConfidence=20
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        items = []
        for label in response.get("ModerationLabels", []):
            items.append(
                ExplicitItem(
                    label=label.get("Name"),
                    likelihood=standardized_confidence_score(label.get("Confidence") / 100),
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        standardized_response = ExplicitContentDataClass(
            items=items, nsfw_likelihood=nsfw_likelihood
        )

        return ResponseType[ExplicitContentDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def image__face_recognition__create_collection(
        self, collection_id: str
    ) -> FaceRecognitionCreateCollectionDataClass:
        client = self.clients["image"]
        try:
            client.create_collection(CollectionId=collection_id)
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        return FaceRecognitionCreateCollectionDataClass(collection_id=collection_id)

    def image__face_recognition__list_collections(
        self,
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        client = self.clients["image"]
        try:
            response = client.list_collections()
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        return ResponseType(
            original_responer=response,
            standardized_response=FaceRecognitionListCollectionsDataClass(
                collections=response["CollectionIds"]
            ),
        )

    def image__face_recognition__list_faces(
        self, collection_id: str
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        client = self.clients["image"]
        try:
            response = client.list_faces(CollectionId=collection_id)
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        face_ids = [face["FaceId"] for face in response["Faces"]]
        # TODO handle NextToken if response is paginated
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionListFacesDataClass(
                face_ids=face_ids
            ),
        )

    def image__face_recognition__delete_collection(
            self, collection_id: str
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        client = self.clients["image"]
        try:
            response = client.delete_collection(CollectionId=collection_id)
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            )
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: BufferedReader
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        client = self.clients["image"]
        file_content = file.read()
        try:
            response = client.index_faces(
                CollectionId=collection_id, Image={"Bytes": file_content}
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        face_ids = [face["Face"]["FaceId"] for face in response["FaceRecords"]]
        if len(face_ids) == 0:
            raise ProviderException("No face detected in the image")

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionAddFaceDataClass(face_ids=face_ids),
        )

    def image__face_recognition__delete_face(
            self, collection_id, face_id
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        client = self.clients["image"]
        try:
            response = client.delete_faces(
                CollectionId=collection_id,
                FaceIds=[
                    face_id,
                ],
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionDeleteFaceDataClass(
                deleted=True
            )
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: BufferedReader
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        client = self.clients["image"]
        file_content = file.read()

        # First check that collection is not empty
        list_faces = self.image__face_recognition__list_faces(collection_id)
        if len(list_faces.standardized_response.face_ids) == 0:
            raise ProviderException("Face Collection is empty.")

        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id, Image={"Bytes": file_content}
            )
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        faces = [
            FaceRecognitionRecognizedFaceDataClass(
                confidence=face["Similarity"] / 100, face_id=face["Face"]["FaceId"]
            )
            for face in response["FaceMatches"]
        ]

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionRecognizeDataClass(items=faces),
        )
