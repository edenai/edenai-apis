import json
import mimetypes
import os
import tempfile
from enum import Enum
from http import HTTPStatus
from json import JSONDecodeError
from typing import Optional

import requests

from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.http import HTTPMethod
from .models import ResponseData


class Parser(Enum):
    RESUME = "resume"
    JD = "job_description"
    INVOICE = "invoice"  # Coming Soon
    RECEIPT = "receipt"  # Coming Soon
    INDOID = "indonesian_id"  # Coming Soon
    AADHAAR = "aadhaar"  # Coming Soon


class Client:
    BASE_URL = "https://service.senseloaf.com"
    __api_key: Optional[str]
    __last_api_response: Optional[dict]
    __last_api_response_type: Optional[str]
    __last_api_response_code: Optional[str]

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        if all([i == "" for i in [api_key, email, password]]):
            raise ProviderException(
                "Please provide api_key or email and password for authentication"
            )
        if api_key == "" and (email == "" or password == ""):
            raise ProviderException(
                "Please provide both email and password for authentication"
            )
        if api_key:
            self.__api_key = api_key.replace("Bearer ", "")
        else:
            self.__login(email, password)

    def __request(
        self,
        method: HTTPMethod,
        url: str,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        json_field: Optional[dict] = None,
        files: Optional[dict] = None,
        params: Optional[dict] = None,
        return_type: Optional[str] = "json",
    ) -> ResponseData:
        response: requests.Response = requests.request(
            method=method.value,
            url=url,
            data=data,
            params=params,
            files=files,
            headers=headers,
            json=json_field,
        )

        try:
            response.raise_for_status()

            if response.status_code == HTTPStatus.NO_CONTENT:
                return ResponseData(
                    response={},
                    response_code=str(response.status_code),
                    response_type="json",
                )

            if return_type == "headers":
                self.__last_api_response = response.headers
                self.__last_api_response_type = "headers"
                self.__last_api_response_code = str(response.status_code)
                return ResponseData(
                    response=response.headers,
                    response_code=str(response.status_code),
                    response_type="headers",
                )
            elif return_type == "content":
                self.__last_api_response = response.content
                self.__last_api_response_type = "content"
                self.__last_api_response_code = str(response.status_code)
                return ResponseData(
                    response={"content": response.content},
                    response_code=str(response.status_code),
                    response_type="content",
                )
            elif return_type == "text":
                self.__last_api_response = response.text
                self.__last_api_response_type = "text"
                self.__last_api_response_code = str(response.status_code)
                return ResponseData(
                    response={"text": response.text},
                    response_code=str(response.status_code),
                    response_type="text",
                )
            else:
                self.__last_api_response = response.json()
                self.__last_api_response_type = "json"
                self.__last_api_response_code = str(response.status_code)
                return ResponseData(
                    response=response.json(),
                    response_code=str(response.status_code),
                    response_type="json",
                )
        except requests.exceptions.HTTPError as exc:
            try:
                message = json.loads(exc.response.text)
                if message.get("errorCode") == "AUTHENTICATION_FAILED":
                    message["errorMessage"] = (
                        "Authentication Failed. Please check your EmailID/Password Combination or API Key"
                    )
                    message["response"]["error"]["faultDetail"] = [
                        "Authentication Failed. Please check your EmailID/Password Combination or API Key"
                    ]
                else:
                    message = exc.response.text
                raise ProviderException(
                    message=f"{exc}\nError message: {message}",
                    code=str(response.status_code),
                ) from exc
            except JSONDecodeError as exp:
                raise ProviderException(
                    message="Internal server error", code=str(response.status_code)
                ) from exp
        except JSONDecodeError:
            raise ProviderException(
                message="Internal server error", code=str(response.status_code)
            )

    def __login(self, email, password):
        url = f"{self.BASE_URL}/login"
        headers = {
            "Content-Type": "application/json",
        }
        payload = json.dumps({"emailId": email, "password": password})
        response = self.__request(
            method=HTTPMethod.POST,
            url=url,
            data=payload,
            headers=headers,
            json_field=None,
            files=None,
            params=None,
            return_type="headers",
        )
        self.__api_key = response.response.get("Authorization").replace("Bearer ", "")

    def __parse_resume_from_file(self, file: str) -> ResponseData:
        url = f"{self.BASE_URL}/api/v2/parse-resume"
        with open(file, "rb") as file_:
            files = [
                (
                    "files",
                    (file.split("/")[-1], file_, mimetypes.guess_type(file)[0]),
                )
            ]
            headers = {"Authorization": f"Bearer {self.__api_key}"}
            response = self.__request(
                method=HTTPMethod.POST,
                url=url,
                data=None,
                headers=headers,
                json_field=None,
                files=files,
                params=None,
                return_type="json",
            )
        if response.response_code != "200":
            raise ProviderException(
                message=response.response, code=response.response_code
            )

        else:
            errors = response.response.get("errors", [])
            if len(errors) > 0:
                raise ProviderException(message=errors[0], code=response.response_code)
            else:
                return response

    def __parse_resume_from_url(self, url: str) -> ResponseData:
        parser_url = f"{self.BASE_URL}/api/v2/parse-resume-url"
        data = json.dumps({"resumeUrl": url})
        headers = {"Authorization": f"Bearer {self.__api_key}"}
        response = self.__request(
            method=HTTPMethod.POST,
            url=parser_url,
            data=data,
            headers=headers,
            json_field=None,
            files=None,
            params=None,
            return_type="json",
        )
        if str(response.response_code) != 200:
            raise ProviderException(
                message=response.response.get("message"), code=response.response_code
            )

        else:
            errors = response.response.get("errors", [])
            if len(errors) > 0:
                raise ProviderException(
                    message=errors[0].get("message"), code=errors[0].get("code")
                )
            else:
                return response

    def __parse_jd_from_file(self, file: str) -> ResponseData:
        url = f"{self.BASE_URL}/api/parse-jd"
        files = [("files", (file, open(file, "rb"), "application/pdf"))]
        headers = {"Authorization": f"Bearer {self.__api_key}"}
        response = self.__request(
            method=HTTPMethod.POST,
            url=url,
            data=None,
            headers=headers,
            json_field=None,
            files=files,
            params=None,
            return_type="json",
        )
        if str(response.response_code) != 200:
            raise ProviderException(
                message=response.response.get("message"), code=response.response_code
            )
        else:
            errors = response.response.get("errors", [])
            if len(errors) > 0:
                raise ProviderException(
                    message=errors[0].get("message"), code=errors[0].get("code")
                )
            else:
                return response

    def __parse_jd_from_url(self, url: str) -> ResponseData:
        tempdir = tempfile.gettempdir()
        filename = url.split("/")[-1]
        filepath = os.path.join(tempdir, filename)
        with open(filepath, "wb") as f:
            f.write(requests.get(url).content)
        return self.__parse_jd_from_file(filepath)

    def __parse_resume(
        self, file: Optional[str] = "", url: Optional[str] = ""
    ) -> ResponseData:
        if file == "" and url == "":
            raise ProviderException(
                "Please provide path to file or url for parsing resume"
            )
        elif file != "":
            return self.__parse_resume_from_file(file)
        elif url != "":
            return self.__parse_resume_from_url(url)

    def __parse_jd(self, file: Optional[str] = None, url: Optional[str] = None):
        if file == "" and url == "":
            raise ProviderException(
                "Please provide path to file or url for parsing resume"
            )
        elif file != "":
            return self.__parse_jd_from_file(file)
        elif url != "":
            return self.__parse_jd_from_url(url)

    def parse_document(
        self, parse_type: Parser, file: Optional[str] = "", url: Optional[str] = ""
    ) -> ResponseData:
        if parse_type.value == "resume":
            return self.__parse_resume(file, url)
        elif parse_type.value == "job_description":
            return self.__parse_jd(file, url)
        elif parse_type.value == "invoice":
            raise NotImplementedError(
                "Invoice parsing is not implemented yet. Reach out to us at team@senseloaf.com for requesting early release"
            )
        elif parse_type.value == "receipt":
            raise NotImplementedError(
                "Receipt parsing is not implemented yet. Reach out to us at team@senseloaf.com for requesting early release"
            )
        elif parse_type.value == "indonesian_id":
            raise NotImplementedError(
                "Indonesian ID card parsing is not implemented yet. Reach out to us at team@senseloaf.com for requesting early release"
            )
        elif parse_type.value == "aadhaar":
            raise NotImplementedError(
                "Indian Aadhaar card parsing is not implemented yet. Reach out to us at team@senseloaf.com for requesting early release"
            )
        else:
            raise NotImplementedError(
                f"Parsing type {parse_type} is not implemented yet. Reach out to us at team@senseloaf.com for requesting early release"
            )

    @property
    def cache(self):
        return ResponseData(
            response=self.__last_api_response,
            response_code=self.__last_api_response_code,
            response_type=self.__last_api_response_type,
        )

    def clear_cache(self):
        self.__last_api_response = {}
        self.__last_api_response_code = ""
        self.__last_api_response_type = ""
