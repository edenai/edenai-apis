from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.image import (
    AutomlClassificationCreateProject,
    AutomlClassificationUploadImage,
    AutomlClassificationRemoveImage,
    AutomlClassificationPrediction,
)


class NyckelCustomImageClassificationApi(ImageInterface):
    def _create_label_if_no_exists(
        self, project_id: str, label_name: str, label_description: str = ""
    ) -> bool:
        self._refresh_session_auth_headers_if_needed()

        url = f"https://www.nyckel.com/v1/functions/{project_id}/labels"
        payload = {"name": label_name, "description": label_description}
        response = self._session.post(url, json=payload)

        if response.status_code >= 400:
            raise self._raise_provider_exception(url, payload, response)

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
