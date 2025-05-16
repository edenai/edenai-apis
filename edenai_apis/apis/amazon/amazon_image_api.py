import json
import base64
from io import BytesIO
from typing import Literal, Optional, Sequence
from edenai_apis.llmengine.utils.moderation import moderate
from edenai_apis.apis.amazon.helpers import handle_amazon_call, get_confidence_if_true
from edenai_apis.features.image.embeddings.embeddings_dataclass import (
    EmbeddingsDataClass,
    EmbeddingDataClass,
)
from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_compare.face_compare_dataclass import (
    FaceCompareDataClass,
    FaceMatch,
    FaceCompareBoundingBox,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceItem,
    FaceLandmarks,
    FaceMakeup,
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
from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class AmazonImageApi(ImageInterface):
    def image__object_detection(
        self, file: str, model: str = None, file_url: str = "", **kwargs
    ) -> ResponseType[ObjectDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        # Getting API response
        payload = {"Image": {"Bytes": file_content}, "MinConfidence": 70}
        original_response = handle_amazon_call(
            self.clients["image"].detect_labels, **payload
        )

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
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        # Getting Response
        payload = {"Image": {"Bytes": file_content}, "Attributes": ["ALL"]}
        original_response = handle_amazon_call(
            self.clients["image"].detect_faces, **payload
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
                eyes_open=get_confidence_if_true(face, "EyeOpen"),
                smile=get_confidence_if_true(face, "Smile"),
                mouth_open=get_confidence_if_true(face, "MouthOpen"),
            )

            # accessories
            accessories = FaceAccessories(
                sunglasses=get_confidence_if_true(face, "Sunglasses"),
                eyeglasses=get_confidence_if_true(face, "Eyeglasses"),
                reading_glasses=None,
                swimming_goggles=None,
                face_mask=None,
                headwear=None,
            )

            # facial hair
            facial_hair = FaceFacialHair(
                moustache=get_confidence_if_true(face, "Mustache"),
                beard=get_confidence_if_true(face, "Beard"),
                sideburns=None,
            )

            # quality
            quality = FaceQuality(
                brightness=face.get("Quality").get("Brightness", 0.0) / 100,
                sharpness=face.get("Quality").get("Sharpness", 0.0) / 100,
                noise=None,
                exposure=None,
                blur=None,
            )

            # emotions
            emotion_output = {}
            for emo in face.get("Emotions", []):
                normalized_emo = emo.get("Confidence", 0.0) * 100
                if emo.get("Type"):
                    if emo["Type"].lower() == "happy":  # normalise keywords
                        emo["Type"] = "happiness"
                    emotion_output[emo["Type"].lower()] = standardized_confidence_score(
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
                unknown=None,
                neutral=None,
                contempt=None,
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
                    occlusions=FaceOcclusions.default(),
                    makeup=FaceMakeup.default(),
                    hair=FaceHair.default(),
                )
            )

        standardized_response = FaceDetectionDataClass(items=faces_list)
        return ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        payload = {"Image": {"Bytes": file_content}, "MinConfidence": 20}
        response = handle_amazon_call(
            self.clients["image"].detect_moderation_labels, **payload
        )

        items = []
        for label in response.get("ModerationLabels", []):
            classificator = CategoryType.choose_category_subcategory(label.get("Name"))
            items.append(
                ExplicitItem(
                    label=label.get("Name"),
                    category=classificator["category"],
                    subcategory=classificator["subcategory"],
                    likelihood=standardized_confidence_score(
                        label.get("Confidence") / 100
                    ),
                    likelihood_score=label.get("Confidence") / 100,
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        nsfw_likelihood_score = (
            ExplicitContentDataClass.calculate_nsfw_likelihood_score(items)
        )
        standardized_response = ExplicitContentDataClass(
            items=items,
            nsfw_likelihood=nsfw_likelihood,
            nsfw_likelihood_score=nsfw_likelihood_score,
        )

        return ResponseType[ExplicitContentDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def image__face_recognition__create_collection(
        self, collection_id: str, **kwargs
    ) -> FaceRecognitionCreateCollectionDataClass:
        payload = {"CollectionId": collection_id}
        handle_amazon_call(self.clients["image"].create_collection, **payload)

        return FaceRecognitionCreateCollectionDataClass(collection_id=collection_id)

    def image__face_recognition__list_collections(
        self, **kwargs
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        response = handle_amazon_call(self.clients["image"].list_collections)

        return ResponseType[FaceRecognitionListCollectionsDataClass](
            original_response=response,
            standardized_response=FaceRecognitionListCollectionsDataClass(
                collections=response["CollectionIds"]
            ),
        )

    def image__face_recognition__list_faces(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        payload = {"CollectionId": collection_id}
        response = handle_amazon_call(self.clients["image"].list_faces, **payload)

        face_ids = [face["FaceId"] for face in response["Faces"]]
        # TODO handle NextToken if response is paginated
        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionListFacesDataClass(face_ids=face_ids),
        )

    def image__face_recognition__delete_collection(
        self, collection_id: str, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        payload = {"CollectionId": collection_id}
        response = handle_amazon_call(
            self.clients["image"].delete_collection, **payload
        )

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionDeleteCollectionDataClass(
                deleted=True
            ),
        )

    def image__face_recognition__add_face(
        self, collection_id: str, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        payload = {"CollectionId": collection_id, "Image": {"Bytes": file_content}}
        response = handle_amazon_call(self.clients["image"].index_faces, **payload)

        face_ids = [face["Face"]["FaceId"] for face in response["FaceRecords"]]
        if len(face_ids) == 0:
            raise ProviderException("No face detected in the image")

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionAddFaceDataClass(face_ids=face_ids),
        )

    def image__face_recognition__delete_face(
        self, collection_id, face_id, **kwargs
    ) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        payload = {"CollectionId": collection_id, "FaceIds": [face_id]}
        response = handle_amazon_call(self.clients["image"].delete_faces, **payload)

        return ResponseType(
            original_response=response,
            standardized_response=FaceRecognitionDeleteFaceDataClass(deleted=True),
        )

    def image__face_recognition__recognize(
        self, collection_id: str, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        client = self.clients["image"]
        with open(file, "rb") as file_:
            file_content = file_.read()

        # First check that collection is not empty
        list_faces = self.image__face_recognition__list_faces(collection_id)
        if len(list_faces.standardized_response.face_ids) == 0:
            raise ProviderException("Face Collection is empty.")

        payload = {"CollectionId": collection_id, "Image": {"Bytes": file_content}}
        response = handle_amazon_call(
            self.clients["image"].search_faces_by_image, **payload
        )

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

    def image__face_compare(
        self, file1: str, file2: str, file1_url: str = "", file2_url: str = "", **kwargs
    ) -> ResponseType[FaceCompareDataClass]:
        client = self.clients.get("image")

        image_source = {}
        image_tar = {}

        with open(file1, "rb") as file1_:
            file1_content = file1_.read()
            image_source["Bytes"] = file1_content

        with open(file2, "rb") as file2_:
            file2_content = file2_.read()
            image_tar["Bytes"] = file2_content

        try:
            response = client.compare_faces(
                SourceImage=image_source,
                TargetImage=image_tar,
            )
        except Exception as excp:
            raise ProviderException(str(excp), code=400)

        face_match_list = []
        for face_match in response.get("FaceMatches", []):
            position = face_match["Face"]["BoundingBox"]
            similarity = face_match.get("Similarity") or 0
            bounding_box = FaceCompareBoundingBox(
                top=position["Top"],
                left=position["Left"],
                height=position["Height"],
                width=position["Width"],
            )
            face_match_obj = FaceMatch(
                confidence=similarity / 100, bounding_box=bounding_box
            )
            face_match_list.append(face_match_obj)

        return ResponseType(
            original_response=response,
            standardized_response=FaceCompareDataClass(items=face_match_list),
        )

    @moderate
    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        # Headers for the HTTP request
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request
        height, width = resolution.split("x")
        model_name, quality = model.split("_")
        request_body = json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": text},
                "imageGenerationConfig": {
                    "numberOfImages": num_images,
                    "quality": quality,
                    "height": int(height),
                    "width": int(width),
                    # "cfgScale": float,
                    # "seed": int
                },
            }
        )

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"amazon.{model_name}",
            "accept": accept_header,
            "contentType": content_type_header,
        }
        response = handle_amazon_call(
            self.clients["bedrock"].invoke_model, **request_params
        )
        response_body = json.loads(response.get("body").read())
        generated_images = []
        for image in response_body["images"]:
            base64_bytes = image.encode("ascii")
            image_bytes = BytesIO(base64.b64decode(base64_bytes))
            resource_url = upload_file_bytes_to_s3(image_bytes, ".png", USER_PROCESS)
            generated_images.append(
                GeneratedImageDataClass(image=image, image_resource_url=resource_url)
            )

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=GenerationDataClass(items=generated_images),
        )

    def image__embeddings(
        self,
        file: str,
        model: str = "titan-embed-image-v1",
        embedding_dimension: Optional[int] = 256,
        representation: Optional[str] = "image",
        file_url: str = "",
        **kwargs,
    ) -> ResponseType[EmbeddingsDataClass]:
        accept_header = "application/json"
        content_type_header = "application/json"

        with open(file, "rb") as image_file:
            image_bytes = image_file.read()

        request_body = {
            "inputImage": base64.b64encode(image_bytes).decode("utf-8"),
            "embeddingConfig": {"outputEmbeddingLength": embedding_dimension},
        }

        request_params = {
            "body": json.dumps(request_body).encode("utf-8"),
            "modelId": f"amazon.{model}",
            "accept": accept_header,
            "contentType": content_type_header,
        }

        response = handle_amazon_call(
            self.clients["bedrock"].invoke_model, **request_params
        )

        original_response = json.loads(response.get("body").read())

        embeddings = original_response["embedding"] or []
        items: Sequence[EmbeddingDataClass] = []
        items.append(EmbeddingDataClass(embedding=embeddings))

        standardized_response = EmbeddingsDataClass(items=items)

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
