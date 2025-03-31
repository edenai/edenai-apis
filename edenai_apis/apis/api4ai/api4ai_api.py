import base64
from io import BytesIO
from json import JSONDecodeError
from typing import Dict, Sequence, Optional, Any

import requests

from edenai_apis.features import ProviderInterface, ImageInterface, OcrInterface
from edenai_apis.features.image.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
    AnonymizationItem,
    AnonymizationBoundingBox,
)
from edenai_apis.features.image.background_removal import BackgroundRemovalDataClass
from edenai_apis.features.image.explicit_content import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.face_detection import (
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceItem,
    FaceLandmarks,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.features.image.logo_detection import (
    LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoVertice,
    LogoItem,
)
from edenai_apis.features.image.object_detection import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.features.ocr.ocr.ocr_dataclass import Bounding_box, OcrDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3, USER_PROCESS
from .helpers import get_errors_from_response
from .types import Api4aiBackgroundRemovalParams


class Api4aiApi(
    ProviderInterface,
    ImageInterface,
    OcrInterface,
):
    provider_name = "api4ai"

    def __init__(self, api_keys: Optional[Dict] = None) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["key"]
        self.base_url = "https://api4ai.cloud"

        self.endpoints = {
            "object_detection": "/general-det/v1/results",
            "logo_detection": "/brand-det/{model}/results",
            "face_detection": "/face-analyzer/v1/results",
            "anonymization": "/img-anonymization/v1/results",
            "nsfw": "/nsfw/v1/results",
            "ocr": "/ocr/v1/results",
            "bg_removal": "/img-bg-removal/v1/general/results",
        }

        self.urls = {
            k: self.base_url + v + f"?api_key={self.api_key}"
            for k, v in self.endpoints.items()
        }

    def image__object_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        This function is used to detect objects in an image.
        """
        with open(file, "rb") as file_:
            files = {"image": file_}
            response = requests.post(self.urls["object_detection"], files=files)
            original_response = response.json()

        if "failure" in original_response["results"][0]["status"]["code"]:
            raise ProviderException(
                original_response["results"][0]["status"]["message"],
                code=response.status_code,
            )

        items = []
        for item in original_response["results"][0]["entities"][0]["objects"]:
            label = next(iter(item.get("entities", [])[0].get("classes", {})))
            confidence = item["entities"][0]["classes"][label]
            if confidence > 0.3:
                boxes = item.get("box", [])
                items.append(
                    ObjectItem(
                        label=label,
                        confidence=confidence,
                        x_min=boxes[0],
                        x_max=boxes[2] + boxes[0],
                        y_min=boxes[1],
                        y_max=boxes[3] + boxes[1],
                    )
                )

        standardized_response = ObjectDetectionDataClass(items=items)
        result = ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__face_detection(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_:
            payload = {
                "image": file_,
            }
            # Get response
            response = requests.post(self.urls["face_detection"], files=payload)
            original_response = response.json()

        # Handle errors
        if "failure" in original_response["results"][0]["status"]["code"]:
            raise ProviderException(
                original_response["results"][0]["status"]["message"],
                code=response.status_code,
            )

        # Face std
        faces_list = []
        faces = original_response["results"][0]["entities"][0]["objects"]
        for face in faces:
            bouding_box = FaceBoundingBox(
                x_min=face["box"][0],
                x_max=face["box"][0],
                y_min=face["box"][0],
                y_max=face["box"][0],
            )
            confidence = face["entities"][0]["classes"]["face"]

            # Landmarks
            landmarks_output = face["entities"][1]["namedpoints"]
            landmarks = FaceLandmarks(
                left_eye=landmarks_output["left-eye"],
                right_eye=landmarks_output["right-eye"],
                nose_tip=landmarks_output["nose-tip"],
                mouth_left=landmarks_output["mouth-left-corner"],
                mouth_right=landmarks_output["mouth-right-corner"],
            )
            faces_list.append(
                FaceItem(
                    confidence=confidence,
                    bounding_box=bouding_box,
                    landmarks=landmarks,
                    emotions=FaceEmotions.default(),
                    poses=FacePoses.default(),
                    age=None,
                    gender=None,
                    hair=FaceHair.default(),
                    facial_hair=FaceFacialHair.default(),
                    quality=FaceQuality.default(),
                    makeup=FaceMakeup.default(),
                    occlusions=FaceOcclusions.default(),
                    accessories=FaceAccessories.default(),
                    features=FaceFeatures.default(),
                )
            )
        standardized_response = FaceDetectionDataClass(items=faces_list)
        result = ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__anonymization(
        self, file: str, file_url: str = "", **kwargs
    ) -> ResponseType[AnonymizationDataClass]:
        with open(file, "rb") as file_:
            files = {"image": file_}
            response = requests.post(self.urls["anonymization"], files=files)

            original_response = response.json()

        if "failure" in original_response["results"][0]["status"]["code"]:
            raise ProviderException(
                original_response["results"][0]["status"]["message"],
                code=response.status_code,
            )

        img_b64 = original_response["results"][0]["entities"][0]["image"]
        entities = original_response["results"][0]["entities"][1].get("objects", [])
        items = []
        for entity in entities:
            for key, value in entity["entities"][0]["classes"].items():
                items.append(
                    AnonymizationItem(
                        kind=key,
                        confidence=value,
                        bounding_boxes=AnonymizationBoundingBox(
                            x_min=entity["box"][0],
                            x_max=entity["box"][1],
                            y_min=entity["box"][2],
                            y_max=entity["box"][3],
                        ),
                    )
                )
        image_data = img_b64.encode()
        image_content = BytesIO(base64.b64decode(image_data))
        resource_url = upload_file_bytes_to_s3(image_content, ".jpeg", USER_PROCESS)
        standardized_response = AnonymizationDataClass(
            image=img_b64, items=items, image_resource_url=resource_url
        )
        result = ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__logo_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[LogoDetectionDataClass]:
        with open(file, "rb") as file_:
            payload = {
                "image": file_,
            }
            # Get response
            response = requests.post(
                self.urls["logo_detection"].format(model=model), files=payload
            )
            if response.status_code >= 400:
                error_message = ""
                try:
                    error_message = response.json().get("message", "")
                except:
                    pass
                error_message = (
                    error_message or "Something went wrong when calling the provider"
                )
                raise ProviderException(error_message, code=response.status_code)

            original_response = response.json()
        # Handle errors
        if "failure" in original_response["results"][0]["status"]["code"]:
            raise ProviderException(
                original_response["results"][0]["status"]["message"],
                code=response.status_code,
            )

        # Get result
        items: Sequence[LogoItem] = []
        logos = original_response["results"][0]["entities"][0]
        if logos.get("strings"):
            for logo in logos.get("strings"):
                items.append(LogoItem(description=logo, bounding_poly=None, score=None))

        if logos.get("objects"):
            for logo in logos.get("objects"):
                brand = logo.get("entities")[0].get("classes")
                try:
                    brand_name, score = list(brand.items())[0]
                except IndexError:
                    continue
                vertices = []
                vertices.append(LogoVertice(x=logo["box"][0], y=logo["box"][1]))
                vertices.append(LogoVertice(x=logo["box"][2], y=logo["box"][1]))
                vertices.append(LogoVertice(x=logo["box"][2], y=logo["box"][3]))
                vertices.append(LogoVertice(x=logo["box"][0], y=logo["box"][3]))
                items.append(
                    LogoItem(
                        description=brand_name,
                        score=score,
                        bounding_poly=LogoBoundingPoly(vertices=vertices),
                    )
                )
        standardized = LogoDetectionDataClass(items=items)
        result = ResponseType[LogoDetectionDataClass](
            original_response=original_response, standardized_response=standardized
        )
        return result

    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            payload = {
                "image": file_,
            }
            # Get response
            response = requests.post(self.urls["nsfw"], files=payload)
            try:
                original_response = response.json()
            except JSONDecodeError as exp:
                raise ProviderException(
                    message="Internal server error", code=response.status_code
                ) from exp

        # Handle errors
        if (
            response.status_code != 200
            or "failure" in original_response["results"][0]["status"]["code"]
        ):
            raise ProviderException(
                response.json()["results"][0]["status"]["message"],
                code=response.status_code,
            )

        # Get result
        nsfw_items = []
        nsfw_response = original_response["results"][0]["entities"][0]["classes"]
        for classe in nsfw_response:
            classificator = CategoryType.choose_category_subcategory(classe)
            nsfw_items.append(
                ExplicitItem(
                    label=classe,
                    category=classificator["category"],
                    subcategory=classificator["subcategory"],
                    likelihood=standardized_confidence_score(nsfw_response[classe]),
                    likelihood_score=nsfw_response[classe],
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(nsfw_items)
        nsfw_likelihood_score = (
            ExplicitContentDataClass.calculate_nsfw_likelihood_score(nsfw_items)
        )
        standardized_response = ExplicitContentDataClass(
            items=nsfw_items,
            nsfw_likelihood=nsfw_likelihood,
            nsfw_likelihood_score=nsfw_likelihood_score,
        )

        result = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        with open(file, "rb") as file_:
            response = requests.post(self.urls["ocr"], files={"image": file_})

        error = get_errors_from_response(response)
        if error is not None:
            raise ProviderException(error, code=response.status_code)

        data = response.json()

        final_text = ""
        boxes: Sequence[Bounding_box] = []

        entities = data["results"][0]["entities"][0]["objects"]
        full_text = ""
        for text in entities:
            box = Bounding_box(
                text=text["entities"][0]["text"],
                top=text["box"][0],
                left=text["box"][1],
                width=text["box"][2],
                height=text["box"][3],
            )
            full_text += text["entities"][0]["text"]
            boxes.append(box)

        final_text += " " + full_text

        standardized_response = OcrDataClass(text=full_text, bounding_boxes=boxes)
        result = ResponseType[OcrDataClass](
            original_response=data,
            standardized_response=standardized_response,
        )
        return result

    def image__background_removal(
        self,
        file: str,
        file_url: str = "",
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[BackgroundRemovalDataClass]:
        if provider_params is None or not isinstance(provider_params, dict):
            api4ai_params = Api4aiBackgroundRemovalParams()
        else:
            api4ai_params = Api4aiBackgroundRemovalParams(**provider_params)

        url: str = self.urls["bg_removal"] + f"&mode={api4ai_params.mode}"
        with open(file, "rb") as f:
            response = requests.post(url, files={"image": f.read()})

            error = get_errors_from_response(response)
            if error is not None:
                raise ProviderException(error, code=response.status_code)

            original_response = response.json()
            img_b64 = original_response["results"][0]["entities"][0]["image"]
            img_fmt = original_response["results"][0]["entities"][0]["format"]

            resource_url = BackgroundRemovalDataClass.generate_resource_url(
                img_b64,
                fmt=img_fmt,
            )

            return ResponseType[BackgroundRemovalDataClass](
                original_response=original_response,
                standardized_response=BackgroundRemovalDataClass(
                    image_b64=img_b64,
                    image_resource_url=resource_url,
                ),
            )
