from io import BufferedReader, BytesIO
from typing import Sequence
import requests
from pdf2image.pdf2image import convert_from_bytes
import json

from edenai_apis.features.image.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
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
from edenai_apis.features import ProviderApi, Image, Ocr
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from .helpers import content_processing


class Api4aiApi(
    ProviderApi,
    Image,
    Ocr,
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
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        This function is used to detect objects in an image.
        """
        files = {"image": file}
        original_response = requests.post(
            self.urls["object_detection"], files=files
        ).json()
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

        standarized_response = ObjectDetectionDataClass(items=items)
        result = ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def image__face_detection(
        self, file: BufferedReader
    ) -> ResponseType[FaceDetectionDataClass]:
        payload = {
            "image": file,
        }
        # Get response
        response = requests.post(self.urls["face_detection"], files=payload)

        # Handle errors
        if response.status_code != 200:
            raise ProviderException(response.json()["result"][0]["status"]["message"])

        # Get result
        original_response = response.json()

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
        standarized_response = FaceDetectionDataClass(items=faces_list)
        result = ResponseType[FaceDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def image__anonymization(
        self, file: BufferedReader
    ) -> ResponseType[AnonymizationDataClass]:
        files = {"image": file}
        response = requests.post(self.urls["anonymization"], files=files)

        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["result"][0]["status"]["message"],
                code=response.status_code,
            )

        img_b64 = original_response["results"][0]["entities"][0]["image"]

        standarized_response = AnonymizationDataClass(image=img_b64)
        result = ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
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
        # Handle errors
        if response.status_code != 200:
            raise ProviderException(response.json()["result"][0]["status"]["message"])

        # Get result
        original_response = response.json()
        logos = original_response["results"][0]["entities"][0]["objects"]
        items: Sequence[LogoItem] = []
        for logo in logos:
            classes = logo.get("entities")[0].get("classes")
            for name in object:
                description = name
                score = classes[name]
            vertices = []
            vertices.append(LogoVertice(logo["box"][0], logo["box"][1]))
            vertices.append(LogoVertice(logo["box"][2], logo["box"][1]))
            vertices.append(LogoVertice(logo["box"][2], logo["box"][3]))
            vertices.append(LogoVertice(logo["box"][0], logo["box"][3]))
            items.append(
                LogoItem(
                    description=description,
                    score=score,
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )
        standarized = LogoDetectionDataClass(items=items)
        result = ResponseType[LogoDetectionDataClass](
            original_response=original_response, standarized_response=standarized
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

        # Handle errors
        if response.status_code != 200:
            raise ProviderException(response.json()["result"][0]["status"]["message"])

        # Get result
        original_response = response.json()
        nsfw_items = []
        nsfw_response = original_response["results"][0]["entities"][0]["classes"]
        for classe in nsfw_response:
            nsfw_items.append(
                ExplicitItem(
                    label=classe, likelihood=content_processing(nsfw_response[classe])
                )
            )
        standarized_response = ExplicitContentDataClass(items=nsfw_items)
        result = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )
        return result

    def ocr__ocr(self, file: BufferedReader, language: str) -> ResponseType[OcrDataClass]:
        payload = {
            "image": file,
        }

        file_content = file.read()
        is_pdf = file.name.lower().endswith(".pdf")
        responses = []

        if is_pdf:
            ocr_file_images = convert_from_bytes(
                file_content, fmt="jpeg", poppler_path=None
            )
            for ocr_image in ocr_file_images:
                ocr_image_buffer = BytesIO()
                ocr_image.save(ocr_image_buffer, format="JPEG")
                ocr_image_buffer.seek(0)

                response = requests.post(self.urls["ocr"], files= {"image" : BufferedReader(ocr_image_buffer)})
                if response.status_code == 200 and not "failure" in response.json()["results"][0]["status"]["code"]:
                    response = response.json()
                    responses.append(response)
            if len(responses) == 0:
                raise ProviderException(f"Can not convert the given file: {file.name}")
                
        else:
            file.seek(0)
            response = requests.post(self.urls["ocr"], files=payload)
            if response.status_code != 200 or "failure" in response.json()["results"][0]["status"]["code"]:
                raise ProviderException(response.json()["result"][0]["status"]["message"])
            response = response.json()
            responses.append(response)


        final_text = ""
        output_value = json.dumps(responses, ensure_ascii=False)
        messages_list = json.loads(output_value)
        boxes: Sequence[Bounding_box] = []

        for original_response in messages_list:
            # original_response = response.json()
            entities = original_response["results"][0]["entities"][0]["objects"]
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

        standarized_response = OcrDataClass(text=full_text, bounding_boxes=boxes).dict()
        result = ResponseType[OcrDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result
