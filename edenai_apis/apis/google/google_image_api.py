from io import BufferedReader
from typing import Optional, Sequence

import numpy as np
from edenai_apis.apis.google.google_helpers import score_to_content
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceItem,
    FaceLandmarks,
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
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from PIL import Image as Img

from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from google.protobuf.json_format import MessageToDict


class GoogleImageApi(ImageInterface):
    def image__explicit_content(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        try:
            response = self.clients["image"].safe_search_detection(image=image)
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

        # Convert response to dict
        data = AnnotateImageResponse.to_dict(response)

        if data.get("error") is not None:
            raise ProviderException(data["error"])

        original_response = data.get("safe_search_annotation", {})

        items = []
        for safe_search_annotation, likelihood in original_response.items():
            items.append(
                ExplicitItem(
                    label=safe_search_annotation.capitalize(), likelihood=likelihood
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(items)

        return ResponseType(
            original_response=original_response,
            standardized_response=ExplicitContentDataClass(
                items=items, nsfw_likelihood=nsfw_likelihood
            ),
        )

    def image__object_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:

        file_ = open(file, "rb")
        image = vision.Image(content=file_.read())
        response = self.clients["image"].object_localization(image=image)
        response = MessageToDict(response._pb)
        file_.close()
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
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        img_size = Img.open(file).size
        image = vision.Image(content=file_content)
        response = self.clients["image"].face_detection(image=image, max_results=100)
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
            )

            # quality
            quality = FaceQuality(
                exposure=2
                * score_to_content(face.get("underExposedLikelihood", 0))
                / 10,
                blur=2 * score_to_content(face.get("blurredLikelihood", 0)) / 10,
            )

            # accessories
            headwear = 2 * score_to_content(face.get("headwearLikelihood", 0)) / 10
            accessories = FaceAccessories(headwear=headwear)

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
                )
            )
        return ResponseType[FaceDetectionDataClass](
            original_response=original_result,
            standardized_response=FaceDetectionDataClass(items=result),
        )

    def image__landmark_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[LandmarkDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)
        response = self.clients["image"].landmark_detection(image=image)
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
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[LogoDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        try:
            response = self.clients["image"].logo_detection(image=image)
        except Exception as provider_call_exception:
            raise ProviderException(str(provider_call_exception))

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
                    LogoVertice(x=float_or_none(vertice.get("x")),
                                y=float_or_none(vertice.get("y")))
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
