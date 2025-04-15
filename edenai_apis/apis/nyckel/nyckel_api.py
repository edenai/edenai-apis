import base64
import json
import time
import uuid
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from edenai_apis.apis.nyckel.nyckel_helpers import check_webhook_result
from edenai_apis.features import ImageInterface, ProviderInterface
from edenai_apis.features.image.automl_classification.create_project.automl_classification_create_project_dataclass import (
    AutomlClassificationCreateProjectDataClass,
)
from edenai_apis.features.image.automl_classification.delete_project.automl_classification_delete_project_dataclass import (
    AutomlClassificationDeleteProjectDataClass,
)
from edenai_apis.features.image.automl_classification.predict_async.automl_classification_predict_async_dataclass import (
    AutomlClassificationPredictAsyncDataClass,
)
from edenai_apis.features.image.automl_classification.train_async.automl_classification_train_async_dataclass import (
    AutomlClassificationTrainAsyncDataClass,
)
from edenai_apis.features.image.automl_classification.upload_data_async.automl_classification_upload_data_async_dataclass import (
    AutomlClassificationUploadDataAsyncDataClass,
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
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
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
        nyckel_predict_error = "No model available to invoke function"
        error = response.text

        if nyckel_predict_error in error:
            error = "Please check your submited data for training/testing"
        raise ProviderException(error, response.status_code)

    def image__search__create_project(self, project_name: str, **kwargs) -> str:
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
        self, file: str, image_name: str, project_id: str, file_url: str = "", **kwargs
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
        self, image_name: str, project_id: str, **kwargs
    ) -> ResponseType[SearchGetImageDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples?externalId={image_name}"
        response = self._session.get(url)
        if not response.status_code == 200:
            self._raise_provider_exception(url, {}, response)

        # The response 'data' key points to a url where we can fetch the image.
        try:
            fetch_image_response = requests.get(response.json()[0]["data"])
            if fetch_image_response.status_code >= 400:
                self._raise_provider_exception(url, {}, fetch_image_response)
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
        self, project_id: str, **kwargs
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
        self, image_name: str, project_id: str, **kwargs
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
        self,
        project_id: str,
        file: Optional[str] = None,
        file_url: Optional[str] = None,
        n: int = 10,
        **kwargs,
    ) -> ResponseType[SearchDataClass]:
        self._refresh_session_auth_headers_if_needed()

        url = (
            f"https://www.nyckel.com/v0.9/functions/{project_id}/"
            f"search?sampleCount={n}"
        )

        if not file:
            assert file_url, "Either file or file_url must be provided"
            data = {"data": file_url}
            response = self._session.post(url, json=data)
        else:
            with open(file, "rb") as f:
                files = {"data": f}
                data = {}
                response = self._session.post(url, files=files)

        if not response.status_code == 200:
            self._raise_provider_exception(url, data, response)

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
        self, name: Optional[str] = None, **kwargs
    ) -> ResponseType[AutomlClassificationCreateProjectDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = "https://www.nyckel.com/v1/functions"
        data = {"input": "Image", "output": "Classification", "name": name}
        try:
            response = self._session.post(url, json=data)
        except:
            raise ProviderException("Something went wrong !!", 500)
        if response.status_code >= 400:
            self._raise_provider_exception(url, {}, response)
            raise ProviderException(message=response.text, code=response.status_code)
        original_response = response.json()

        standardized_response = AutomlClassificationCreateProjectDataClass(
            project_id=original_response.get("id"), name=original_response.get("name")
        )
        del original_response["id"]
        del original_response["projectId"]
        return ResponseType[AutomlClassificationCreateProjectDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def __image__automl_classification_delete_image(
        self, project_id: str, picture_id: str, **kwargs
    ):
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples/{picture_id}"

        try:
            response = self._session.delete(url)
        except Exception as exc:
            raise ProviderException("Something went wrong !!", 500) from exc

        if response.status_code >= 400:
            self._raise_provider_exception(url, {}, response)

        return True

    def __create_label_if_no_exists(
        self, project_id: str, label_name: str, label_description: str = ""
    ) -> bool:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/labels"
        payload = {"name": label_name, "description": label_description}
        try:
            response = self._session.post(url, json=payload)
            original_response = response.json()
        except:
            raise ProviderException(
                "Something went wrong when creating the label !!", 500
            )
        if (
            response.status_code >= 400
            and "already exists" not in original_response.get("message", "")
        ):
            raise self._raise_provider_exception(url, payload, response)

    def image__automl_classification__upload_data_async__launch_job(
        self,
        project_id: str,
        label: str,
        type_of_data: str,
        file: str,
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/samples"
        file_ = None

        if not label:
            raise ProviderException("Label needs to be specified !!")

        # Create Label
        self.__create_label_if_no_exists(project_id=project_id, label_name=label)

        # Upload Sample
        post_parameters = {"url": url}
        if file_url:
            post_parameters["json"] = {
                "annotation": {"LabelName": label},
                "data": file_url,
            }
        else:
            file_ = open(file, "rb")
            post_parameters["files"] = {"data": file_}
            post_parameters["data"] = {"annotation.labelName": label}

        response = self._session.post(**post_parameters)
        if file_ is not None:
            file_.close()
        if response.status_code >= 400:
            # self._raise_provider_exception(url, post_parameters, response)
            self.handle_provider_error(response)
        try:
            original_response = response.json()
        except Exception as exp:
            raise ProviderException("Something went wrong !!", 500) from exp

        data = original_response
        data["label_name"] = label
        job_id = str(uuid.uuid4())
        data_job_id = {job_id: data}
        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[400, 404, 408, 429, 500, 502, 503, 504],
        )  # retry in case of failure. Because if it fails, the image will be loaded but for the user it's a fail
        s.mount("http://", HTTPAdapter(max_retries=retries))
        try:
            webook_response = s.post(
                url=f"https://webhook.site/{self.webhook_token}",
                data=json.dumps(data_job_id),
                headers={"content-type": "application/json"},
            )
        except Exception as exp:
            self.__image__automl_classification_delete_image(project_id, data["id"])
            raise ProviderException("Could not upload image data", 400) from exp
        if webook_response.status_code >= 400:
            self.__image__automl_classification_delete_image(project_id, data["id"])
            raise ProviderException("Could not upload image data", 400)
        return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def image__automl_classification__upload_data_async__get_job_result(
        self, provider_job_id: str, **kwargs
    ) -> AsyncBaseResponseType[AutomlClassificationUploadDataAsyncDataClass]:
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
            return AsyncPendingResponseType[
                AutomlClassificationUploadDataAsyncDataClass
            ](provider_job_id=provider_job_id)
        return AsyncResponseType[AutomlClassificationUploadDataAsyncDataClass](
            original_response=original_response,
            standardized_response=AutomlClassificationUploadDataAsyncDataClass(
                message="Data uploaded successfully",
                image=original_response.get("data"),
                label_name=original_response.get("label_name"),
            ),
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__train_async__launch_job(
        self, project_id: str, **kwargs
    ) -> AsyncLaunchJobResponseType:
        return AsyncLaunchJobResponseType(provider_job_id=project_id)

    def image__automl_classification__train_async__get_job_result(
        self, provider_job_id: str, **kwargs
    ) -> AsyncBaseResponseType[AutomlClassificationTrainAsyncDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{provider_job_id}/samples"
        try:
            response = self._session.get(url)
        except:
            return AsyncResponseType[AutomlClassificationTrainAsyncDataClass](
                original_response="",
                standardized_response=AutomlClassificationTrainAsyncDataClass(
                    message="Model is trained", project_id=provider_job_id, name=None
                ),
                provider_job_id=provider_job_id,
            )
        response_json = response.json()
        if response.status_code >= 400:
            return AsyncResponseType[AutomlClassificationTrainAsyncDataClass](
                original_response="",
                standardized_response=AutomlClassificationTrainAsyncDataClass(
                    message="Model is trained", project_id=provider_job_id, name=None
                ),
                provider_job_id=provider_job_id,
            )
        if len(response_json) < 4:
            raise ProviderException(
                message="There must 2 images per labels and at least 2 labels in the project"
            )
        labels = [
            res["annotation"]["labelId"]
            for res in response_json
            if res.get("annotation")
        ]
        unique_labels = set(labels)
        nb_two_labes = 0
        for x in unique_labels:
            if labels.count(x) >= 2:
                nb_two_labes += 1
                if nb_two_labes >= 2:
                    break
        if nb_two_labes < 2:
            raise ProviderException(message="Each label must have at least 2 samples")
        return AsyncResponseType[AutomlClassificationTrainAsyncDataClass](
            original_response="",
            standardized_response=AutomlClassificationTrainAsyncDataClass(
                message="Model is trained", project_id=provider_job_id, name=None
            ),
            provider_job_id=provider_job_id,
        )

    def image__automl_classification__predict_async__launch_job(
        self, project_id: str, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}/invoke"
        if file_url != "":
            data = {"data": file_url}
            try:
                response = self._session.post(url, json=data)
            except:
                raise ProviderException(
                    "Something went wrong when running the prediction !!", 500
                )
        else:
            with open(file, "rb") as f:
                data = {"data": f}
                try:
                    response = self._session.post(url, files=data)
                except:
                    raise ProviderException(
                        "Something went wrong when running the prediction !!", 500
                    )
        if response.status_code >= 400:
            self._raise_provider_exception(url, {}, response)
        job_id = str(uuid.uuid4())
        original_response = response.json()
        standardized_response = AutomlClassificationPredictAsyncDataClass(
            label=original_response.get("labelName", ""),
            confidence=original_response.get("confidence", 0),
        )
        return AsyncResponseType[AutomlClassificationPredictAsyncDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=job_id,
        )

    def image__automl_classification__delete_project(
        self, project_id: str, **kwargs
    ) -> ResponseType[AutomlClassificationDeleteProjectDataClass]:
        self._refresh_session_auth_headers_if_needed()
        url = f"https://www.nyckel.com/v1/functions/{project_id}"
        try:
            response = self._session.delete(url)
        except:
            ProviderException("Something went wrong when deleting the project")
        if response.status_code >= 400:
            raise ProviderException(
                message=(
                    response.text
                    if response.text != ""
                    else "This project does not exist"
                ),
                code=response.status_code,
            )
        return ResponseType[AutomlClassificationDeleteProjectDataClass](
            original_response="",
            standardized_response=AutomlClassificationDeleteProjectDataClass(
                deleted=True
            ),
        )

    def handle_provider_error(self, response: requests.Response):
        """
        402	Billing issue. Mostly likely because you have exceeded the free tier quota.
        403	Forbidden. Check your credentials.
        409	Resouce conflict. Commonly raised when trying to create a sample that already exists in the function (Nyckel does not allow duplicate samples). When annotating an existing sample, use the PUT samples endpoint instead.
        429	Throttled. You have exceeded either 25 requests per second or 25 concurrent requests.
        500	Internal error. Retry -- ideally with exponential backoff.
        503	Service temporarily unavailable. Retry -- ideally with exponential backoff.
        """
        if response.status_code == 402:
            raise ProviderException(
                "Billing issue. Mostly likely because you have exceeded the free tier quota.",
                response.status_code,
            )
        elif response.status_code == 403:
            raise ProviderException("Forbidden. Check your credentials.")
        elif response.status_code == 409:
            raise ProviderException(
                "Resouce conflict. Commonly raised when trying to create a sample that already exists in the function (Nyckel does not allow duplicate samples). When annotating an existing sample, use the PUT samples endpoint instead.",
                response.status_code,
            )
        elif response.status_code == 429:
            raise ProviderException(
                "Throttled. You have exceeded either 25 requests per second or 25 concurrent requests.",
                response.status_code,
            )
        elif response.status_code == 500:
            raise ProviderException(
                "Internal error. Retry -- ideally with exponential backoff.",
                response.status_code,
            )
        elif response.status_code == 503:
            raise ProviderException(
                "Service temporarily unavailable. Retry -- ideally with exponential backoff.",
                response.status_code,
            )
        else:
            raise ProviderException(
                f"Unexpected error with status code {response.status_code}: {response.text}",
                response.status_code,
            )
