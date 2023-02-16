from io import BufferedReader
from typing import Sequence
import requests
from pdf2image.pdf2image import convert_from_bytes

from edenai_apis.features.image.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
    AnonymizationItem,
    AnonymizationBoundingBox
)
from edenai_apis.features.image.explicit_content import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_detection import (
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceItem,
    FaceLandmarks,
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
from edenai_apis.features import ProviderInterface, ImageInterface, OcrInterface
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from .helpers import get_errors_from_response


class Api4aiApi(
    ProviderInterface,
    ImageInterface,
    OcrInterface,
):

    provider_name = "api4ai"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["key"]
        self.urls = {
            "object_detection": f"{self.api_settings['object_detection']['url']}"
                + f"?api_key={self.api_key}",
            "logo_detection": f"{self.api_settings['logo_detection']['url']}"
                + f"?api_key={self.api_key}",
            "face_detection": f"{self.api_settings['face_detection']['url']}"
                + f"?api_key={self.api_key}",
            "anonymization": "https://api4ai.cloud/img-anonymization/v1/results"
                + f"?api_key={self.api_key}",
            "nsfw": f"https://api4ai.cloud/nsfw/v1/results?api_key={self.api_key}",
            "ocr": f"https://api4ai.cloud/ocr/v1/results?api_key={self.api_key}",
        }

    def image__object_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        This function is used to detect objects in an image.
        """

        file_ = open(file, "rb")
        files = {"image": file_}
        original_response = requests.post(
            self.urls["object_detection"], files=files
        ).json()

        file_.close()
        
        if 'failure' in original_response['results'][0]['status']['code']:
            raise ProviderException(original_response['results'][0]['status']['message'])
            
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
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:

        file_ = open(file, "rb")
        payload = {
            "image": file_,
        }
        # Get response
        response = requests.post(self.urls["face_detection"], files=payload)
        original_response = response.json()
        file_.close()
        
        # Handle errors
        if 'failure' in original_response['results'][0]['status']['code']:
            raise ProviderException(original_response['results'][0]['status']['message'])

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
                    confidence=confidence, bounding_box=bouding_box, landmarks=landmarks
                )
            )
        standardized_response = FaceDetectionDataClass(items=faces_list)
        result = ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__anonymization(
        self, file: BufferedReader
    ) -> ResponseType[AnonymizationDataClass]:
        files = {"image": file}
        response = requests.post(self.urls["anonymization"], files=files)

        original_response = response.json()

        if 'failure' in original_response['results'][0]['status']['code']:
            raise ProviderException(original_response['results'][0]['status']['message'])

        img_b64 = original_response["results"][0]["entities"][0]["image"]
        entities = original_response["results"][0]["entities"][1].get("objects", [])
        items = []
        for entity in entities:
            for key, value in entity["entities"][0]["classes"].items():
                items.append(AnonymizationItem(
                    kind=key,
                    confidence=value,
                    bounding_boxes=AnonymizationBoundingBox(x_min=entity["box"][0], x_max=entity["box"][1], y_min=entity["box"][2], y_max=entity["box"][3]),
                ))
            

        standardized_response = AnonymizationDataClass(image=img_b64, items=items)
        result = ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__logo_detection(
        self, file: BufferedReader
    ) -> ResponseType[LogoDetectionDataClass]:
        payload = {
            "image": file,
        }
        # Get response
        response = requests.post(self.urls["logo_detection"], files=payload)
        original_response = response.json()
        # Handle errors
        if 'failure' in original_response['results'][0]['status']['code']:
            raise ProviderException(original_response['results'][0]['status']['message'])

        # Get result
        logos = original_response["results"][0]["entities"][0]["objects"]
        items: Sequence[LogoItem] = []
        for logo in logos:
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
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        payload = {
            "image": file,
        }
        # Get response
        response = requests.post(self.urls["nsfw"], files=payload)
        original_response = response.json()

        # Handle errors
        if response.status_code != 200 or "failure" in original_response["results"][0]["status"]["code"]:
            raise ProviderException(response.json()["results"][0]["status"]["message"])
        
        # Get result
        nsfw_items = []
        nsfw_response = original_response["results"][0]["entities"][0]["classes"]
        for classe in nsfw_response:
            nsfw_items.append(
                ExplicitItem(
                    label=classe, likelihood=standardized_confidence_score(nsfw_response[classe])
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(nsfw_items)
        standardized_response = ExplicitContentDataClass(items=nsfw_items, nsfw_likelihood=nsfw_likelihood)

        result = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
        return result

    def ocr__ocr(self, file: BufferedReader, language: str) -> ResponseType[OcrDataClass]:

        response = requests.post(self.urls["ocr"], files={"image": file})

        error = get_errors_from_response(response)
        if error is not None:
            raise ProviderException(error)

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

        standardized_response = OcrDataClass(text=full_text, bounding_boxes=boxes).dict()
        result = ResponseType[OcrDataClass](
            original_response=data,
            standardized_response=standardized_response,
        )
        return result
