import base64
import json
from typing import Sequence, Optional
import numpy as np
import mimetypes
import requests
from PIL import Image as Img, UnidentifiedImageError
from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from google.protobuf.json_format import MessageToDict

from edenai_apis.apis.google.google_helpers import (
    handle_google_call,
    score_to_content,
    get_access_token,
)
from edenai_apis.features.image.explicit_content.category import CategoryType
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
    FaceHair,
    FaceItem,
    FaceLandmarks,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.image.landmark_detection.landmark_detection_dataclass import (
    LandmarkDetectionDataClass,
    LandmarkItem,
    LandmarkLatLng,
    LandmarkLocation,
    LandmarkVertice,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoItem,
    LogoVertice,
)
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.features.image.question_answer import QuestionAnswerDataClass
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.parsing import extract
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.image.embeddings import (
    EmbeddingsDataClass,
    EmbeddingDataClass,
)


class GoogleImageApi(ImageInterface):
    def _convert_likelihood(self, value: int) -> float:
        values = [0, 0.2, 0.4, 0.6, 0.8, 1]
        return values[value]

    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        payload = {"image": image}
        response = handle_google_call(
            self.clients["image"].safe_search_detection, **payload
        )

        # Convert response to dict
        data = AnnotateImageResponse.to_dict(response)

        if data.get("error") is not None:
            raise ProviderException(data["error"])

        original_response = data.get("safe_search_annotation", {})

        items = []
        for safe_search_annotation, likelihood in original_response.items():
            classificator = CategoryType.choose_category_subcategory(
                safe_search_annotation.capitalize()
            )
            items.append(
                ExplicitItem(
                    label=safe_search_annotation.capitalize(),
                    category=classificator["category"],
                    subcategory=classificator["subcategory"],
                    likelihood_score=self._convert_likelihood(likelihood),
                    likelihood=likelihood,
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(items)
        nsfw_likelihood_score = (
            ExplicitContentDataClass.calculate_nsfw_likelihood_score(items)
        )
        return ResponseType(
            original_response=original_response,
            standardized_response=ExplicitContentDataClass(
                items=items,
                nsfw_likelihood=nsfw_likelihood,
                nsfw_likelihood_score=nsfw_likelihood_score,
            ),
        )

    def image__object_detection(
        self, file: str, model: str = None, file_url: str = "", **kwargs
    ) -> ResponseType[ObjectDetectionDataClass]:
        with open(file, "rb") as file_:
            image = vision.Image(content=file_.read())

            payload = {"image": image}
            response = handle_google_call(
                self.clients["image"].object_localization, **payload
            )
            response = MessageToDict(response._pb)

        items = []
        for object_annotation in response.get("localizedObjectAnnotations", []):
            x_min, x_max = np.infty, -np.infty
            y_min, y_max = np.infty, -np.infty
            # Getting borders
            for normalize_vertice in object_annotation["boundingPoly"][
                "normalizedVertices"
            ]:
                x_min, x_max = min(x_min, normalize_vertice.get("x", 0)), max(
                    x_max, normalize_vertice.get("x", 0)
                )
                y_min, y_max = min(y_min, normalize_vertice.get("y", 0)), max(
                    y_max, normalize_vertice.get("y", 0)
                )
            items.append(
                ObjectItem(
                    label=object_annotation["name"],
                    confidence=object_annotation["score"],
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        return ResponseType[ObjectDetectionDataClass](
            original_response=response,
            standardized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        try:
            img_size = Img.open(file).size
        except UnidentifiedImageError:
            raise ProviderException(message="Can not identify image file", code=400)
        image = vision.Image(content=file_content)

        payload = {"image": image, "max_results": 100}
        response = handle_google_call(self.clients["image"].face_detection, **payload)
        original_result = MessageToDict(response._pb)

        result = []
        width, height = img_size
        for face in original_result.get("faceAnnotations", []):
            # emotions
            emotions = FaceEmotions(
                joy=score_to_content(face.get("joyLikelihood")),
                sorrow=score_to_content(face.get("sorrowLikelihood")),
                anger=score_to_content(face.get("angerLikelihood")),
                surprise=score_to_content(face.get("surpriseLikelihood")),
                # Not supported by Google
                # ------------------------
                disgust=None,
                fear=None,
                confusion=None,
                calm=None,
                contempt=None,
                unknown=None,
                neutral=None,
                # ------------------------
            )

            # quality
            quality = FaceQuality(
                exposure=2
                * score_to_content(face.get("underExposedLikelihood", 0))
                / 10,
                blur=2 * score_to_content(face.get("blurredLikelihood", 0)) / 10,
                noise=None,
                brightness=None,
                sharpness=None,
            )

            # accessories
            accessories = FaceAccessories.default()
            accessories.headwear = (
                2 * score_to_content(face.get("headwearLikelihood", 0)) / 10
            )

            # landmarks
            landmark_output = {}
            for land in face.get("landmarks", []):
                if "type" in land and "UNKNOWN_LANDMARK" not in land:
                    landmark_output[land["type"]] = [
                        land["position"]["x"] / width,
                        land["position"]["y"] / height,
                    ]
            landmarks = FaceLandmarks(
                left_eye=landmark_output.get("LEFT_EYE", []),
                left_eye_top=landmark_output.get("LEFT_EYE_TOP_BOUNDARY", []),
                left_eye_right=landmark_output.get("LEFT_EYE_RIGHT_CORNER", []),
                left_eye_bottom=landmark_output.get("LEFT_EYE_BOTTOM_BOUNDARY", []),
                left_eye_left=landmark_output.get("LEFT_EYE_LEFT_CORNER", []),
                right_eye=landmark_output.get("RIGHT_EYE", []),
                right_eye_top=landmark_output.get("RIGHT_EYE_TOP_BOUNDARY", []),
                right_eye_right=landmark_output.get("RIGHT_EYE_RIGHT_CORNER", []),
                right_eye_bottom=landmark_output.get("RIGHT_EYE_BOTTOM_BOUNDARY", []),
                right_eye_left=landmark_output.get("RIGHT_EYE_LEFT_CORNER", []),
                left_eyebrow_left=landmark_output.get("LEFT_OF_LEFT_EYEBROW", []),
                left_eyebrow_right=landmark_output.get("LEFT_OF_RIGHT_EYEBROW", []),
                left_eyebrow_top=landmark_output.get("LEFT_EYEBROW_UPPER_MIDPOINT", []),
                right_eyebrow_left=landmark_output.get("RIGHT_OF_LEFT_EYEBROW", []),
                right_eyebrow_right=landmark_output.get("RIGHT_OF_RIGHT_EYEBROW", []),
                nose_tip=landmark_output.get("NOSE_TIP", []),
                nose_bottom_right=landmark_output.get("NOSE_BOTTOM_RIGHT", []),
                nose_bottom_left=landmark_output.get("NOSE_BOTTOM_LEFT", []),
                mouth_left=landmark_output.get("MOUTH_LEFT", []),
                mouth_right=landmark_output.get("MOUTH_RIGHT", []),
                right_eyebrow_top=landmark_output.get(
                    "RIGHT_EYEBROW_UPPER_MIDPOINT", []
                ),
                midpoint_between_eyes=landmark_output.get("MIDPOINT_BETWEEN_EYES", []),
                nose_bottom_center=landmark_output.get("NOSE_BOTTOM_CENTER", []),
                upper_lip=landmark_output.get("GET_UPPER_LIP", []),
                under_lip=landmark_output.get("GET_LOWER_LIP", []),
                mouth_center=landmark_output.get("MOUTH_CENTER", []),
                left_ear_tragion=landmark_output.get("LEFT_EAR_TRAGION", []),
                right_ear_tragion=landmark_output.get("RIGHT_EAR_TRAGION", []),
                forehead_glabella=landmark_output.get("FOREHEAD_GLABELLA", []),
                chin_gnathion=landmark_output.get("CHIN_GNATHION", []),
                chin_left_gonion=landmark_output.get("CHIN_LEFT_GONION", []),
                chin_right_gonion=landmark_output.get("CHIN_RIGHT_GONION", []),
                left_cheek_center=landmark_output.get("LEFT_CHEEK_CENTER", []),
                right_cheek_center=landmark_output.get("RIGHT_CHEEK_CENTER", []),
            )

            # bounding box
            bounding_poly = face.get("fdBoundingPoly", {}).get("vertices", [])

            result.append(
                FaceItem(
                    accessories=accessories,
                    quality=quality,
                    emotions=emotions,
                    landmarks=landmarks,
                    poses=FacePoses(
                        roll=face.get("rollAngle"),
                        pitch=face.get("panAngle"),
                        yaw=face.get("tiltAngle"),
                    ),
                    confidence=face.get("detectionConfidence"),
                    # indices are this way because array of bounding boxes
                    # follow this pattern:
                    # [top-left, top-right, bottom-right, bottom-left]
                    bounding_box=FaceBoundingBox(
                        x_min=bounding_poly[0].get("x", 0.0) / width,
                        x_max=bounding_poly[1].get("x", width) / width,
                        y_min=bounding_poly[0].get("y", 0.0) / height,
                        y_max=bounding_poly[3].get("y", height) / height,
                    ),
                    # Not supported by Google Cloud Vision
                    # --------------------
                    age=None,
                    gender=None,
                    hair=FaceHair.default(),
                    facial_hair=FaceFacialHair.default(),
                    makeup=FaceMakeup.default(),
                    occlusions=FaceOcclusions.default(),
                    features=FaceFeatures.default(),
                    # --------------------
                )
            )
        return ResponseType[FaceDetectionDataClass](
            original_response=original_result,
            standardized_response=FaceDetectionDataClass(items=result),
        )

    def image__landmark_detection(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[LandmarkDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)
        payload = {"image": image}
        response = handle_google_call(
            self.clients["image"].landmark_detection, **payload
        )
        dict_response = vision.AnnotateImageResponse.to_dict(response)
        landmarks = dict_response.get("landmark_annotations", [])

        items: Sequence[LandmarkItem] = []
        for landmark in landmarks:
            if landmark.get("description") not in [item.description for item in items]:
                vertices: Sequence[LandmarkVertice] = []
                for poly in landmark.get("bounding_poly", {}).get("vertices", []):
                    vertices.append(LandmarkVertice(x=poly["x"], y=poly["y"]))
                locations = []
                for location in landmark.get("locations", []):
                    locations.append(
                        LandmarkLocation(
                            lat_lng=LandmarkLatLng(
                                latitude=location.get("lat_lng", {}).get("latitude"),
                                longitude=location.get("lat_lng", {}).get("longitude"),
                            )
                        )
                    )
                items.append(
                    LandmarkItem(
                        description=landmark.get("description"),
                        confidence=landmark.get("score"),
                        bounding_box=vertices,
                        locations=locations,
                    )
                )
        if dict_response.get("error"):
            raise ProviderException(
                message=dict_response["error"].get(
                    "message", "Error calling Google Api"
                )
            )

        return ResponseType[LandmarkDetectionDataClass](
            original_response=landmarks,
            standardized_response=LandmarkDetectionDataClass(items=items),
        )

    def image__logo_detection(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[LogoDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        payload = {"image": image}
        response = handle_google_call(self.clients["image"].logo_detection, **payload)

        response = MessageToDict(response._pb)

        float_or_none = lambda val: float(val) if val else None
        # Handle error
        if response.get("error", {}).get("message") is not None:
            raise ProviderException(response["error"]["message"])

        items: Sequence[LogoItem] = []
        for key in response.get("logoAnnotations", []):
            vertices = []
            for vertice in key.get("boundingPoly").get("vertices"):
                vertices.append(
                    LogoVertice(
                        x=float_or_none(vertice.get("x")),
                        y=float_or_none(vertice.get("y")),
                    )
                )

            items.append(
                LogoItem(
                    description=key.get("description"),
                    score=key.get("score"),
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )
        return ResponseType[LogoDetectionDataClass](
            original_response=response,
            standardized_response=LogoDetectionDataClass(items=items),
        )

    def image__question_answer(
        self,
        file: str,
        temperature: float,
        max_tokens: int,
        file_url: str = "",
        model: Optional[str] = None,
        question: Optional[str] = None,
        settings: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[QuestionAnswerDataClass]:
        with open(file, "rb") as fstream:
            file_content = fstream.read()
            file_b64 = base64.b64encode(file_content).decode("utf-8")
        mime_type = mimetypes.guess_type(file)[0]
        image_data = f"data:{mime_type};base64,{file_b64}"
        response = self.clients["llm_client"].image_qa(
            image_data=image_data,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            question=question,
        )
        return response

    def image__embeddings(
        self,
        file: Optional[str],
        representation: Optional[str] = "image",
        model: Optional[str] = "multimodalembedding@001",
        embedding_dimension: int = 1408,
        file_url: Optional[str] = "",
        **kwargs,
    ) -> ResponseType[EmbeddingsDataClass]:

        token = get_access_token(self.location)
        location = "us-central1"
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{location}/publishers/google/models/{model}:predict"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        with open(file, "rb") as file_:
            content = file_.read()
            payload = {
                "instances": [
                    {
                        "image": {
                            "bytesBase64Encoded": base64.b64encode(content).decode(
                                "utf-8"
                            )
                        }
                    }
                ],
                "parameters": {"dimension": embedding_dimension},
            }

            response = requests.post(url, json=payload, headers=headers)
            try:
                original_response = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException(
                    message="Internal Server Error",
                    code=500,
                ) from exc

            if "error" in original_response:
                raise ProviderException(
                    message=original_response["error"]["message"], code=400
                )

            if not original_response.get("predictions"):
                raise ProviderException(message="No predictions found", code=400)

            items: Sequence[EmbeddingDataClass] = []

            for prediction in original_response["predictions"]:
                embedding = prediction.get("imageEmbedding") or []
                items.append(EmbeddingDataClass(embedding=embedding))

            standardized_response = EmbeddingsDataClass(items=items)

            return ResponseType[EmbeddingsDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )
