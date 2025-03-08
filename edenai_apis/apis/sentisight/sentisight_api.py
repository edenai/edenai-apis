import base64
from typing import Dict, Sequence, Optional, Any

import requests
from PIL import Image as Img

from edenai_apis.features import ProviderInterface, OcrInterface, ImageInterface
from edenai_apis.features.image import (
    SearchDataClass,
    ImageItem,
    ObjectDetectionDataClass,
    ObjectItem,
    ExplicitContentDataClass,
    ExplicitItem,
    BackgroundRemovalDataClass,
)
from edenai_apis.features.image.explicit_content.category import CategoryType
from edenai_apis.features.image.search.get_image import (
    SearchGetImageDataClass,
)
from edenai_apis.features.image.search.get_images import (
    ImageSearchItem,
    SearchGetImagesDataClass,
)
from edenai_apis.features.image.search.upload_image.search_upload_image_dataclass import (
    SearchUploadImageDataClass,
)
from edenai_apis.features.image.search.delete_image.search_delete_image_dataclass import (
    SearchDeleteImageDataClass,
)
from edenai_apis.features.ocr import OcrDataClass, Bounding_box
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import add_query_param_in_url
from edenai_apis.utils.exception import ProviderException, LanguageException
from edenai_apis.utils.types import ResponseType, ResponseSuccess
from .sentisight_helpers import (
    calculate_bounding_box,
    get_formatted_language,
    handle_error_image_search,
)
from .types import SentisightBackgroundRemovalParams, SentisightPreTrainModel


class SentiSightApi(ProviderInterface, OcrInterface, ImageInterface):
    provider_name: str = "sentisight"

    def __init__(self, api_keys: Optional[Dict[str, str]] = None) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.key = self.api_settings["auth-token"]
        self.base_url = "https://platform.sentisight.ai/api/pm-predict/"
        self.headers = {"X-Auth-token": self.key, "Content-Type": "application/json"}

    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        url = f"{self.base_url}{SentisightPreTrainModel.TEXT_RECOGNITION.value}"

        if not language:
            raise LanguageException("Language not provided")

        with open(file, "rb") as file_:
            response = requests.post(
                url=add_query_param_in_url(
                    url, {"lang": get_formatted_language(language)}
                ),
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file_,
            )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)
        response = response.json()
        with Img.open(file) as img:
            width, height = img.size
        # response["width"], response["height"] = Img.open(file).size

        bounding_boxes: Sequence[Bounding_box] = []

        text = ""
        for item in response:
            if text == "":
                text = item["label"]
            else:
                text = text + " " + item["label"]
            bounding_box = calculate_bounding_box(item["points"], width, height)
            bounding_boxes.append(
                Bounding_box(
                    text=item["label"],
                    left=float(bounding_box["x"]),
                    top=float(bounding_box["y"]),
                    width=float(bounding_box["width"]),
                    height=float(bounding_box["height"]),
                )
            )

        standardized_response = OcrDataClass(
            text=text.replace("\n", " ").strip(), bounding_boxes=bounding_boxes
        )
        result = ResponseType[OcrDataClass](
            original_response=response,
            standardized_response=standardized_response,
        )
        return result

    def image__object_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ObjectDetectionDataClass]:
        with open(file, "rb") as file_:
            response = requests.post(
                self.base_url + SentisightPreTrainModel.OBJECT_DETECTION.value,
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file_,
            )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        with Img.open(file) as img:
            width = img.width
            height = img.height

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
        standardized_response = ObjectDetectionDataClass(items=objects)
        result = ResponseType[ObjectDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            response = requests.post(
                self.base_url + SentisightPreTrainModel.NSFW_CLASSIFICATION.value,
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file_,
            )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        original_response = response.json()
        items: Sequence[ObjectItem] = []
        items.append(
            ExplicitItem(
                label="nudity",
                likelihood=round(
                    [x for x in original_response if x["label"] == "unsafe"][0]["score"]
                    / 20
                ),
                likelihood_score=[
                    x for x in original_response if x["label"] == "unsafe"
                ][0]["score"]
                / 100,
                category=CategoryType.choose_category_subcategory("nudity")["category"],
                subcategory=CategoryType.choose_category_subcategory("nudity")[
                    "subcategory"
                ],
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

        result = ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def image__search__create_project(self, project_name: str, **kwargs) -> str:
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
            handle_error_image_search(response)

        project_id = response.json()["id"]
        return project_id

    def image__search__upload_image(
        self, file: str, image_name: str, project_id: str, file_url: str = "", **kwargs
    ) -> ResponseType[SearchUploadImageDataClass]:
        upload_project_url = (
            "https://platform.sentisight.ai/api/image/"
            + f"{project_id}/{image_name}?preprocess=true"
        )
        # Build the request
        with open(file, "rb") as file_:
            response = requests.post(
                upload_project_url,
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file_,
            )

        if response.status_code != 200:
            handle_error_image_search(response)

        return ResponseType[SearchUploadImageDataClass](
            standardized_response=SearchUploadImageDataClass(status="success"),
            original_response={},
        )

    def image__search__delete_image(
        self, image_name: str, project_id: str, **kwargs
    ) -> ResponseType[SearchDeleteImageDataClass]:
        delete_project_url = (
            f"https://platform.sentisight.ai/api/image/{project_id}/{image_name}/"
        )

        response = requests.delete(delete_project_url, headers=self.headers, data={})

        if response.status_code != 200:
            handle_error_image_search(response)

        return ResponseType[SearchDeleteImageDataClass](
            standardized_response=SearchDeleteImageDataClass(status="success"),
            original_response={},
        )

    def image__search__get_images(
        self, project_id: str, **kwargs
    ) -> ResponseType[SearchGetImagesDataClass]:
        get_images_url = f"https://platform.sentisight.ai/api/images/{project_id}/"
        response = requests.get(get_images_url, headers=self.headers)

        if response.status_code != 200:
            handle_error_image_search(response)

        images = []
        original_response = list(response.json())
        for image in original_response:
            images.append(ImageSearchItem(image_name=image))
        standardized_response = SearchGetImagesDataClass(list_images=images)
        return ResponseType[SearchGetImagesDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__search__get_image(
        self, image_name: str, project_id: str, **kwargs
    ) -> ResponseType[SearchGetImageDataClass]:
        get_image_url = (
            f"https://platform.sentisight.ai/api/image/{project_id}/{image_name}/"
        )

        # Build the request
        response = requests.get(get_image_url, headers=self.headers, data={})

        # Handle provider error
        if response.status_code != 200:
            handle_error_image_search(response)

        image_b64 = base64.b64encode(response.content)

        image = SearchGetImageDataClass(image=image_b64)
        # Return the image as bytes
        return ResponseType[SearchGetImageDataClass](
            original_response=response.content, standardized_response=image
        )

    def image__search__launch_similarity(
        self,
        project_id: str,
        file: Optional[str] = None,
        file_url: Optional[str] = None,
        n: int = 10,
        **kwargs,
    ) -> ResponseType[SearchDataClass]:
        search_project_url = (
            "https://platform.sentisight.ai/api/similarity"
            + f"?project={project_id}&limit={n}&threshold=0&and=false"
        )
        if not file:
            raise ValueError("file is required.")
        with open(file, "rb") as file_:
            response = requests.post(
                search_project_url,
                headers={
                    "accept": "*/*",
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=file_,
            )

        # Handle the error
        if response.status_code != 200:
            handle_error_image_search(response)
        items = []
        for image in response.json():
            items.append(ImageItem(image_name=image["image"], score=image["score"]))
        standardized_response = SearchDataClass(items=items)
        result = ResponseType[SearchDataClass](
            original_response=response.json(),
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
        with open(file, "rb") as fstream:
            if provider_params is None or not isinstance(provider_params, dict):
                sentisight_params = SentisightBackgroundRemovalParams()
            else:
                sentisight_params = SentisightBackgroundRemovalParams(**provider_params)

            response = requests.post(
                self.base_url + SentisightPreTrainModel.BACKGROUND_REMOVAL.value,
                headers={
                    "X-Auth-token": self.key,
                    "Content-Type": "application/octet-stream",
                },
                data=fstream.read(),
            )

            if response.status_code != 200:
                raise ProviderException(response.text, code=response.status_code)

            original_response = response.json()
            img_b64 = original_response[0]["image"]
            resource_url = BackgroundRemovalDataClass.generate_resource_url(img_b64)

            return ResponseType[BackgroundRemovalDataClass](
                original_response=original_response,
                standardized_response=BackgroundRemovalDataClass(
                    image_b64=img_b64,
                    image_resource_url=resource_url,
                ),
            )
