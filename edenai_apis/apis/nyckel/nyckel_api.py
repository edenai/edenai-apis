import base64
import json
import time
from typing import Dict, Optional

import requests
from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image import (
    AutomlClassificationCreateProject,
    AutomlClassificationUploadImage,
    AutomlClassificationRemoveImage,
    AutomlClassificationPrediction,
)
from edenai_apis.features.image.search.delete_image.search_delete_image_dataclass import (
    SearchDeleteImageDataClass,
)
from edenai_apis.features.image.search.get_image.search_get_image_dataclass import (
    SearchGetImageDataClass,
)
from edenai_apis.features.image.search.get_images.search_get_images_dataclass import (
    ImageSearchItem,
    SearchGetImagesDataClass,
)
from edenai_apis.features.image.search.search_dataclass import (
    ImageItem,
    SearchDataClass,
)
from edenai_apis.features.image.search.upload_image.search_upload_image_dataclass import (
    SearchUploadImageDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseSuccess, ResponseType


def strip_nyckel_prefix(prefixed_id: str) -> str:
    split_id = prefixed_id.split("_")
    if len(split_id) == 2:
        return split_id[1]
    else:
        return prefixed_id


class NyckelApi(ProviderInterface, ImageInterface):
    provider_name: str = "nyckel"
    DEFAULT_SIMILAR_IMAGE_COUNT = 10

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self._session = requests.Session()
        self._renew_at = 0

    def _refresh_session_auth_headers_if_needed(self) -> None:
        if time.time() > self._renew_at:
            self._renew_session_auth_header()

    def _renew_session_auth_header(self) -> None:
        RENEW_MARGIN_SECONDS = 10 * 60

        url = "https://www.nyckel.com/connect/token"
        data = {
            "client_id": self.api_settings["client_id"],
            "client_secret": self.api_settings["client_secret"],
            "grant_type": "client_credentials",
        }

        response = requests.post(url, data=data)
        if not response.status_code == 200:
            self._raise_provider_exception(url, data, response)

        self._session.headers.update(
            {"authorization": "Bearer " + response.json()["access_token"]}
        )
        self._renew_at = (
            time.time() + response.json()["expires_in"] - RENEW_MARGIN_SECONDS
        )

    def _raise_provider_exception(
        self, url: str, data: dict, response: requests.Response
    ) -> None:
        error_message = f"Call to {url=} with payload={data} failed with {response.status_code}: {response.text}."
        raise ProviderException(error_message)

    def image__search__create_project(self, project_name: str) -> str:
        """
        Search by image
        """
        self._refresh_session_auth_headers_if_needed()
        url = "https://www.nyckel.com/v1/functions"
        data = {"input": "Image", "output": "Search", "name": project_name}
        response = self._session.post(url, json=data)
        if not response.status_code == 200:
            self._raise_provider_exception(url, data, response)
        return strip_nyckel_prefix(response.json()["id"])

    def image__search__upload_image(
        self, file: str, image_name: str, project_id: str, file_url: str = ""
    ) -> ResponseType[SearchUploadImageDataClass]:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples"

        if file == "" or file is None:
            assert (
                file_url and file_url != ""
            ), "Either file or file_url must be provided"
            data = {"data": file_url, "externalId": image_name}
            response = self._session.post(url, json=data)
        else:
            with open(file, "rb") as f:
                data = {"externalId": image_name}
                files = {"data": f}
                response = self._session.post(url, files=files, data=data)

        if not response.status_code == 200:
            self._raise_provider_exception(url, data, response)

        return ResponseType[SearchUploadImageDataClass](
            standardized_response=SearchUploadImageDataClass(status="success"),
            original_response=response.json(),
        )

    def image__search__get_image(
        self, image_name: str, project_id: str
    ) -> ResponseType[SearchGetImageDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples?externalId={image_name}"
        response = self._session.get(url)
        if not response.status_code == 200:
            self._raise_provider_exception(url, {}, response)

        # The response 'data' key points to a url where we can fetch the image.
        try:
            fetch_image_response = requests.get(response.json()[0]["data"])
            fetch_image_response.raise_for_status()
        except IndexError:
            raise ProviderException(f"Image '{image_name}' not found.")
        except Exception:
            raise ProviderException(
                f"Unable to fetch image bytes from {response.json()[0]['data']}"
            )

        image_b64 = base64.b64encode(fetch_image_response.content)

        return ResponseType[SearchGetImageDataClass](
            original_response=response.json(),
            standardized_response=SearchGetImageDataClass(image=image_b64),
        )

    def image__search__get_images(
        self, project_id: str
    ) -> ResponseType[SearchGetImagesDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples?batchSize=1000"
        response = self._session.get(url)
        if not response.status_code == 200:
            self._raise_provider_exception(url, {}, response)

        images = [
            ImageSearchItem(image_name=entry["externalId"]) for entry in response.json()
        ]
        standardized_response = SearchGetImagesDataClass(list_images=images)
        return ResponseType[SearchGetImagesDataClass](
            original_response=response.json(),
            standardized_response=standardized_response,
        )

    def image__search__delete_image(
        self, image_name: str, project_id: str
    ) -> ResponseType[SearchDeleteImageDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples?externalId={image_name}"

        response = self._session.delete(url)

        if response.status_code != 200:
            self._raise_provider_exception(url, {}, response)

        return ResponseType[SearchDeleteImageDataClass](
            original_response=None,
            standardized_response=SearchDeleteImageDataClass(status="success"),
        )

    def image__search__launch_similarity(
        self, project_id: str, file: Optional[str] = None, file_url: str = ""
    ) -> ResponseType[SearchDataClass]:
        self._refresh_session_auth_headers_if_needed()

        url = (
            f"https://www.nyckel.com/v0.9/functions/{project_id}/"
            f"search?sampleCount={self.DEFAULT_SIMILAR_IMAGE_COUNT}"
        )

        if file == "" or file is None:
            assert (
                file_url and file_url != ""
            ), "Either file or file_url must be provided"
            data = {"data": file_url}
            response = self._session.post(url, json=data)
        else:
            with open(file, "rb") as f:
                files = {"data": f}
                data = {}
                response = self._session.post(url, files=files)

        if not response.status_code == 200:
            self._raise_provider_exception(url, data, response)

        print(response.json())
        return ResponseType[SearchDataClass](
            original_response=response.json(),
            standardized_response=SearchDataClass(
                items=[
                    ImageItem(
                        image_name=entry["externalId"],
                        score=1.0 - entry["distance"],
                    )
                    for entry in response.json()["searchSamples"]
                ]
            ),
        )

    def image__automl_classification__create_project(
        self, name: str = ""
    ) -> ResponseType[AutomlClassificationCreateProject]:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions"
        data = {"name": name, "input": "Text", "output": "Classification"}
        try:
            response = self._session.post(url, json=data)
        except:
            raise ProviderException("Something went wrong", code=500)

        if response.status_code >= 400:
            self._raise_provider_exception(url, data, response)

        original_response = response.json()
        project_id = original_response["id"]

        return ResponseType[AutomlClassificationCreateProject](
            original_response=original_response,
            standardized_response=AutomlClassificationCreateProject(
                project_id=project_id
            ),
        )

    def _create_label_if_no_exists(
        self, project_id: str, label_name: str, label_description: str = ""
    ) -> bool:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/labels"
        payload = {"name": label_name, "description": label_description}
        response = self._session.post(url, json=payload)

        if response.status_code >= 400:  # exists already
            raise self._raise_provider_exception(url, payload, response)

    def image__automl_classification__upload_image(
        self, file: str, project_id: str, label: str, file_url: str = ""
    ) -> ResponseType[AutomlClassificationUploadImage]:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples"
        file_ = None

        if not label:
            raise ProviderException("Label needs to be specified !!")
        payload = (
            {"annotation": {"labelName": label}}
            if file_url
            else {"annotation.labelName": label}
        )

        post_parameters = {"url": url}

        if file_url:
            post_parameters["json"] = payload
            payload["data"] = file_url

        else:
            file_ = open(file, "rb")
            post_parameters.update({"files": {"data": file_}, "data": payload})

        try_again = True
        error_message = f"A label with name '{label}' was not found"

        while try_again:  # in case the label is not already created, it will be created
            try:
                response = self._session.post(**post_parameters)
            except:
                raise ProviderException("Something went wrong !!", 500)

            if response.status_code >= 400:
                if (
                    response.json().get("message") == error_message
                ):  # label with the provided name does not exists
                    self._create_label_if_no_exists(project_id, label)
                else:
                    self._raise_provider_exception(url, post_parameters, response)
            if response.status_code == 200:
                try_again = False

        if file:
            file_.close()

        original_response = response.json()
        image_id = original_response["id"]

        return ResponseType[AutomlClassificationUploadImage](
            original_response=original_response,
            standardized_response=AutomlClassificationUploadImage(image_id=image_id),
        )

    def image__automl_classification__remove_image(
        self, project_id: str, picture_id: str
    ) -> ResponseType[AutomlClassificationRemoveImage]:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples/{picture_id}"

        try:
            response = self._session.delete(url)
        except:
            raise ProviderException("Something went wrong !!", 500)

        if response.status_code >= 400:
            self._raise_provider_exception(url, {}, response)

        original_response = response.json()
        return ResponseType[AutomlClassificationRemoveImage](
            original_response=original_response,
            standardized_response=AutomlClassificationRemoveImage(removed=True),
        )

    def image__automl_classification__prediction(
        self, file: str, project_id: str, file_url: str
    ) -> ResponseType[AutomlClassificationPrediction]:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/invoke"

        file_ = None

        post_parameters = {"url": url}

        if file_url:
            post_parameters["json"] = {"data": file_url}
        else:
            file_ = open(file, "rb")
            post_parameters["files"] = {"data": file_}

        try:
            response = self._session.post(**post_parameters)
        except:
            raise ProviderException("Something went wrong !!", 500)

        if response.status_code >= 400:
            self._raise_provider_exception(url, post_parameters, response)

        if file_:
            file_.close()

        original_response = response.json()
        label_name = original_response["labelName"]
        confidence = original_response["confidence"]

        return ResponseType[AutomlClassificationPrediction](
            original_response=original_response,
            standardized_response=AutomlClassificationPrediction(
                label=label_name, confidence=confidence
            ),
        )
