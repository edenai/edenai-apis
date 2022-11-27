from io import BufferedReader, BytesIO
from typing import Sequence
from PIL import Image as Img
import base64
import requests
from pdf2image.pdf2image import convert_from_bytes
import json

from edenai_apis.features import ProviderApi, Ocr, Image
from edenai_apis.features.image import (
    SearchDataClass,
    ImageItem,
    ObjectDetectionDataClass,
    ObjectItem,
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.search.get_images import (
    ImageSearchItem,
    SearchGetImagesDataClass,
)
from edenai_apis.features.image.search.get_image import (
    SearchGetImageDataClass,
)
from edenai_apis.features.ocr import OcrDataClass, Bounding_box
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import format_string_url_language
from edenai_apis.utils.exception import ProviderException, LanguageException
from edenai_apis.utils.types import ResponseType, ResponseSuccess
from .sentisight_helpers import calculate_bounding_box, get_formatted_language


class SentiSightApi(ProviderApi, Ocr, Image):
    provider_name: str = "sentisight"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.key = self.api_settings["auth-token"]
        self.base_url = self.api_settings["baseUrl"]
        self.headers = {"X-Auth-token": self.key, "Content-Type": "application/json"}

    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        url = f"{self.base_url}Text-recognition"
        
        #check language
        if not language:
            raise LanguageException("Language not provided")
        is_pdf = file.name.lower().endswith(".pdf")
        file_content = file.read()
        responses = []
        widths_heights = []
        if is_pdf:
            ocr_file_images = convert_from_bytes(
                file_content, fmt="jpeg", poppler_path=None
            )
            for ocr_image in ocr_file_images:
                ocr_image_buffer = BytesIO()
                ocr_image.save(ocr_image_buffer, format="JPEG")

                # call
                response = requests.post(
                    format_string_url_language(
                        url, get_formatted_language(language), "lang", self.provider_name
                    ),
                    headers={
                        "accept": "*/*",
                        "X-Auth-token": self.key,
                        "Content-Type": "application/octet-stream",
                    },
                    data=ocr_image_buffer.getvalue(),
                )
                if response.status_code == 200:
                    response = response.json()
                    widths_heights.append(ocr_image.size)
                    responses.append(response)
            if len(responses) == 0:
                raise ProviderException(f"Can not convert the given file: {file.name}")

        else:
            file.seek(0)
            response = requests.post(
                format_string_url_language(
                    url, get_formatted_language(language), "lang", self.provider_name
                ),
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file,
           )
            if response.status_code != 200:
                raise ProviderException(response.text)
            response= response.json()
            widths_heights.append(Img.open(file).size)
            # response["width"], response["height"] = Img.open(file).size
            responses.append(response)

        final_text = ""
        output_value = json.dumps(responses, ensure_ascii=False)
        messages_list = json.loads(output_value)
        bounding_boxes: Sequence[Bounding_box] = []

        for response_index in range(len(messages_list)):
            response = messages_list[response_index]
            text = ""
            for item in response:
                if text == "":
                    text = item["label"]
                else:
                    text = text + " " + item["label"]
                width_height = widths_heights[response_index]
                bounding_box = calculate_bounding_box(item["points"], *width_height)
                bounding_boxes.append(
                    Bounding_box(
                        text=item["label"],
                        left=float(bounding_box["x"]),
                        top=float(bounding_box["y"]),
                        width=float(bounding_box["width"]),
                        height=float(bounding_box["height"]),
                    )
                )
            final_text += " " + text
            

        standarized_response = OcrDataClass(text=final_text.replace("\n", " ").strip(), bounding_boxes=bounding_boxes)
        result = ResponseType[OcrDataClass](
            original_response=messages_list,
            standarized_response=standarized_response,
        )
        return result

    def image__object_detection(
        self, file: BufferedReader
    ) -> ResponseType[ObjectDetectionDataClass]:
        response = requests.post(
            self.base_url + "Object-detection",
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
                "Content-Type": "application/octet-stream",
            },
            data=file,
        )
        img = Img.open(file)
        width = img.width
        height = img.height

        if response.status_code != 200:
            raise ProviderException(response.text)

        original_response = response.json()
        objects: Sequence[ObjectItem] = []
        for obj in original_response:
            objects.append(
                ObjectItem(
                    label=obj["label"],
                    confidence=obj["score"] / 100,
                    x_min=float(obj["x0"]) / width,
                    x_max=float(obj["x1"]) / width,
                    y_min=float(obj["y0"] / height),
                    y_max=float(obj["y1"] / height),
                )
            )
        standarized_response = ObjectDetectionDataClass(items=objects)
        result = ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def image__explicit_content(
        self, file: BufferedReader
    ) -> ResponseType[ExplicitContentDataClass]:
        response = requests.post(
            self.base_url + "NSFW-classification",
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
                "Content-Type": "application/octet-stream",
            },
            data=file,
        )
        if response.status_code != 200:
            raise ProviderException(response.text)

        original_response = response.json()
        items: Sequence[ObjectItem] = []
        items.append(
            ExplicitItem(
                label="nudity",
                likelihood=round(
                    [x for x in original_response if x["label"] == "unsafe"][0]["score"]
                    / 20
                ),
            )
        )
        standarized_response = ExplicitContentDataClass(items=items)
        result = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def image__search__create_project(self, project_name: str) -> str:
        create_project_url = "https://platform.sentisight.ai/api/project"

        json_data = {
            "name": project_name,
        }
        response = requests.post(
            create_project_url,
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
            },
            json=json_data,
        )

        if response.status_code != 200:
            raise ProviderException(response.text)

        project_id = response.json()["id"]
        return project_id

    def image__search__upload_image(
        self, file: BufferedReader, image_name: str, project_id: str
    ) -> ResponseSuccess:
        upload_project_url = (
            "https://platform.sentisight.ai/api/image/"
            + f"{project_id}/{image_name}?preprocess=true"
        )
        # Build the request
        response = requests.post(
            upload_project_url,
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
                "Content-Type": "application/octet-stream",
            },
            data=file,
        )

        # Handle response error
        if response.status_code != 200:
            raise ProviderException(response.text)
        return ResponseSuccess()

    def image__search__delete_image(
        self, image_name: str, project_id: str
    ) -> ResponseSuccess:

        delete_project_url = (
            f"https://platform.sentisight.ai/api/image/{project_id}/{image_name}/"
        )

        response = requests.delete(delete_project_url, headers=self.headers, data={})

        if response.status_code != 200:
            raise ProviderException(response.text)
        return ResponseSuccess()

    def image__search__get_images(
        self, project_id: str
    ) -> ResponseType[SearchGetImagesDataClass]:
        get_images_url = f"https://platform.sentisight.ai/api/images/{project_id}/"
        response = requests.get(get_images_url, headers=self.headers)

        if response.status_code != 200:
            raise ProviderException(response.text)

        images = []
        original_response = list(response.json())
        for image in original_response:
            images.append(ImageSearchItem(image_name=image))
        standardized_response = SearchGetImagesDataClass(list_images=images)
        return ResponseType[SearchGetImagesDataClass](
            original_response=original_response,
            standarized_response=standardized_response,
        )

    def image__search__get_image(
        self, image_name: str, project_id: str
    ) -> ResponseType[SearchGetImageDataClass]:
        get_image_url = (
            f"https://platform.sentisight.ai/api/image/{project_id}/{image_name}/"
        )

        # Build the request
        response = requests.get(get_image_url, headers=self.headers, data={})

        # Handle provider error
        if response.status_code != 200:
            raise ProviderException(response.text)

        image_b64 = base64.b64encode(response.content)

        image = SearchGetImageDataClass(image=image_b64)
        # Return the image as bytes
        return ResponseType[SearchGetImageDataClass](
            original_response=response.content,
            standarized_response=image
        )

    def image__search__launch_similarity(
        self, file: BufferedReader, project_id: str
    ) -> ResponseType[SearchDataClass]:
        search_project_url = (
            "https://platform.sentisight.ai/api/similarity"
            + f"?project={project_id}&limit=10&threshold=0&and=false"
        )
        response = requests.post(
            search_project_url,
            headers={
                "accept": "*/*",
                "X-Auth-token": self.key,
                "Content-Type": "application/octet-stream",
            },
            data=file,
        )

        # Handle the error
        if response.status_code != 200:
            raise ProviderException(response.text)

        items = []
        for image in response.json():
            items.append(ImageItem(image_name=image["image"], score=image["score"]))
        standarized_response = SearchDataClass(items=items)
        result = ResponseType[SearchDataClass](
            original_response=response,
            standarized_response=standarized_response,
        )
        return result
