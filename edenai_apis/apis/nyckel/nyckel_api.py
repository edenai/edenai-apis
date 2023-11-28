import base64
import time
import uuid
import json

from typing import Dict, Optional

import requests

from edenai_apis.apis.nyckel.nyckel_helpers import check_webhook_result
from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.automl_classification.create_project_async.automl_classification_create_project_async_dataclass import (
    AutomlClassificationCreateProjectDataClass,
)
from edenai_apis.features.image.automl_classification.delete_project_async.automl_classification_delete_project_async_dataclass import (
    AutomlClassificationDeleteProjectDataClass,
)
from edenai_apis.features.image.automl_classification.predict_async.automl_classification_predict_async_dataclass import (
    AutomlClassificationPredictDataClass,
)
from edenai_apis.features.image.automl_classification.train_async.automl_classification_train_async_dataclass import (
    AutomlClassificationTrainDataClass,
)
from edenai_apis.features.image.automl_classification.upload_data_async.automl_classification_upload_data_async_dataclass import (
    AutomlClassificationUploadDataDataClass,
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
from edenai_apis.utils.types import (
    ResponseType,
    AsyncLaunchJobResponseType,
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    AsyncErrorResponseType,
)


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
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings["webhook_token"]

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
        raise ProviderException(response.text)

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

    def image__automl_classification__create_project_async__launch_job(
        self, name: Optional[str] = None
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = "https://www.nyckel.com/v1/functions"
        data = {"input": "Image", "output": "Classification", "name": name}
        response = self._session.post(url, json=data)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        original_response = response.json()
        return AsyncLaunchJobResponseType(provider_job_id=original_response["id"])

    def image__automl_classification__create_project_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationCreateProjectDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{provider_job_id}"
        response = self._session.get(url)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        original_response = response.json()
        return AsyncResponseType[AutomlClassificationCreateProjectDataClass](
            original_response=original_response,
            standardized_response=AutomlClassificationCreateProjectDataClass(
                status="created",
                name=original_response.get("name", None),
                project_id=provider_job_id,
            ),
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__upload_data_async__launch_job(
        self,
        project_id: str,
        label: str,
        type_of_data: str,
        file: str,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples"
        labels_response = self._session.get(
            f"https://www.nyckel.com/v1/functions/{project_id}/labels"
        )
        if labels_response.status_code != 200:
            raise ProviderException(
                message=labels_response.text, code=labels_response.status_code
            )
        labels = labels_response.json()
        if label not in [l["name"] for l in labels]:
            new_label = self._session.post(
                f"https://www.nyckel.com/v1/functions/{project_id}/labels",
                json={"name": label},
            )
            if new_label.status_code != 200:
                raise ProviderException(
                    message=new_label.text, code=new_label.status_code
                )
        if file_url != "":
            data = {"data": file_url, "annotation": {"labelName": label}}
            response = self._session.post(url, json=data)
            if response.status_code != 200:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )
            print(response.text)
            return AsyncLaunchJobResponseType(provider_job_id=project_id)
        with open(file, "rb") as f:
            data = {"annotation.labelName": label}
            files = {"data": f}
            response = self._session.post(url, files=files, data=data)
            print(response.text)
            if response.status_code != 200:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )
        return AsyncLaunchJobResponseType(provider_job_id=project_id)

    def image__automl_classification__upload_data_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationUploadDataDataClass]:
        return AsyncResponseType[AutomlClassificationUploadDataDataClass](
            original_response=None,
            standardized_response=AutomlClassificationUploadDataDataClass(
                status="uploaded"
            ),
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__train_async__launch_job(
        self, project_id: str
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=project_id)

    def image__automl_classification__train_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationTrainDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{provider_job_id}/samples"
        response = self._session.get(url)
        response_json = response.json()
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        if len(response_json) < 4:
            raise ProviderException(
                message="There must 2 images per labels and at least 2 labels in the project"
            )
        labels = [res["annotation"]["labelId"] for res in response_json]
        unique_labels = set(labels)
        for x in unique_labels:
            if labels.count(x) < 2:
                raise ProviderException(
                    message="Each label must have at least 2 samples"
                )
        return AsyncResponseType[AutomlClassificationTrainDataClass](
            original_response=None,
            standardized_response=AutomlClassificationTrainDataClass(
                status="trained", project_id=provider_job_id, name=None
            ),
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__predict_async__launch_job(
        self, project_id: str, file: str, file_url: str = ""
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/invoke"
        if file_url != "":
            data = {"data": file_url}
            response = self._session.post(url, json=data)
        else:
            with open(file, "rb") as f:
                data = {"data": f}
                response = self._session.post(url, files=data)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        job_id = str(uuid.uuid4())
        data_job_id = {job_id: response.json()}
        requests.post(
            url=f"https://webhook.site/{self.webhook_token}",
            data=json.dumps(data_job_id),
            headers={"content-type": "application/json"},
        )
        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def image__automl_classification__predict_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationPredictDataClass]:
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        webhook_result, response_status = check_webhook_result(
            provider_job_id, self.webhook_settings
        )
        if response_status != 200:
            raise ProviderException(webhook_result, code=response_status)
        result_object = (
            next(
                filter(
                    lambda response: provider_job_id in response["content"],
                    webhook_result,
                ),
                None,
            )
            if webhook_result
            else None
        )
        if not result_object or not result_object.get("content"):
            raise ProviderException("Provider returned an empty response")
        try:
            original_response = json.loads(result_object["content"]).get(
                provider_job_id, None
            )
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")
        if original_response is None:
            return AsyncPendingResponseType[AutomlClassificationPredictDataClass](
                provider_job_id=provider_job_id
            )
        standardized_response = AutomlClassificationPredictDataClass(
            label=original_response.get("labelName", ""),
            confidence=original_response.get("confidence", 0),
        )
        return AsyncResponseType[AutomlClassificationPredictDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__delete_project_async__launch_job(
        self, project_id: str
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}"
        response = self._session.delete(url)
        if response.status_code != 200:
            raise ProviderException(
                message=response.text
                if response.text != ""
                else "This function does not exist",
                code=response.status_code,
            )
        job_id = str(uuid.uuid4())
        data_job_id = {job_id: response.text}
        requests.post(
            url=f"https://webhook.site/{self.webhook_token}",
            data=json.dumps(data_job_id),
            headers={"content-type": "application/json"},
        )
        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def image__automl_classification__delete_project_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationDeleteProjectDataClass]:
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        webhook_result, response_status = check_webhook_result(
            provider_job_id, self.webhook_settings
        )
        if response_status != 200:
            raise ProviderException(webhook_result, code=response_status)
        result_object = (
            next(
                filter(
                    lambda response: provider_job_id in response["content"],
                    webhook_result,
                ),
                None,
            )
            if webhook_result
            else None
        )
        if not result_object or not result_object.get("content"):
            raise ProviderException("Provider returned an empty response")
        try:
            original_response = json.loads(result_object["content"]).get(
                provider_job_id, None
            )
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")
        if original_response is None:
            return AsyncPendingResponseType[AutomlClassificationDeleteProjectDataClass](
                provider_job_id=provider_job_id
            )
        standardized_response = AutomlClassificationDeleteProjectDataClass(deleted=True)
        return AsyncResponseType[AutomlClassificationDeleteProjectDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )
