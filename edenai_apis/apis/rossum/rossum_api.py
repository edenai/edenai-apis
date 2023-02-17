from enum import Enum
from io import BufferedReader
from time import sleep
import requests
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import InvoiceParserDataClass
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class RossumApi(ProviderInterface, OcrInterface):
    provider_name = "rossum"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.username = self.api_settings["username"]
        self.password = self.api_settings["password"]
        self.url = self.api_settings["url"]
        self.queue_id = self.api_settings["queue_id"]

        self._login()

    def _login(self):
        """
        Login to the provider and store the token in token attribute

        Raises:
            ProviderException: If the status code is not 200
        """
        response = requests.post(
            url=self.url + "auth/login",
            json={
                "username": self.username,
                "password": self.password
            },
            headers={
                "Content-Type": "application/json"
            }
        )

        if response.status_code != 200:
            raise ProviderException(message="Error while getting token", code=response.status_code)

        self.token = response.json()["key"]

    class EndpointType(Enum):
        LOGIN = "LOGIN"
        UPLOAD = "UPLOAD"
        DOWNLOAD = "DOWNLOAD"


    def _get_endpoint(self, endpoint_type: EndpointType) -> str:
        """
        Get the endpoint from the endpoint type

        Args:
            endpoint_type (EndpointType): An enum of endpoint type (LOGIN, UPLOAD, STATUS, DOWNLOAD)

        Returns:
            str: The endpoint
        """
        self.endpoints = {
            "LOGIN": "auth/login",
            "UPLOAD": f"queues/{self.queue_id}/upload",
            "DOWNLOAD": f"queues/{self.queue_id}/export"
        }

        return self.url + self.endpoints[endpoint_type.value]

    def _upload(self, file: BufferedReader) -> str:
        """
        Upload a file to the provider

        Args:
            file (BufferedReader): The file to upload

        Returns:
           str: The annotation endpoint of the file

        Raises:
            ProviderException: If an error occurs while uploading the file (Status code != 201)
        """
        response = requests.post(
            url=self._get_endpoint(self.EndpointType.UPLOAD),
            files={
                "content": file
            },
            headers={
                "Authorization": f"Token {self.token}",
            }
        )

        if response.status_code != 201:
            raise ProviderException(message="Error while uploading file", code=response.status_code)

        return response.json()['annotation']

    def _get_status_and_id(self, annotation_endpoint: str) -> tuple:
        """
        Get the status of the file and this id

        Args:
            annotation_endpoint (str): The annotation endpoint of the file

        Returns:
            tuple: The tuple with the id and the status of the file (id, status) (ex: ('122', 'to_review')))

        Raises:
            ProviderException: If an error occurs while checking the status (Status code != 200)
        """
        response = requests.get(
            url=annotation_endpoint,
            headers={
                "Authorization": f"Token {self.token}"
            }
        )

        if response.status_code != 200:
            raise ProviderException(message="Error while checking status", code=response.status_code)

        response_json = response.json()
        return (response_json['id'], response_json['status'])

    def _download_reviewing_data(self, id: str) -> dict:
        """
        Download the reviewing data

        Returns:
            ResponseType[dict]: The reviewing data

        Raises:
            ProviderException: If an error occurs while downloading the reviewing data (Status code != 200)
        """
        response = requests.get(
            url=self._get_endpoint(self.EndpointType.DOWNLOAD) + f"?status=exported&format=json&id={id}",
            headers={
                "Authorization": f"Token {self.token}"
            }
        )
        print(response.status_code)
        print(response.text)
        if response.status_code != 200:
            raise ProviderException(message="Error while downloading reviewing data", code=response.status_code)

        return response.json()

    def ocr__invoice_parser(self, file: BufferedReader, language: str) -> ResponseType[InvoiceParserDataClass]:
        annotation_endpoint = self._upload(file)
        id, status = self._get_status_and_id(annotation_endpoint)
        while status != 'to_review':
            sleep(1)
            id, status = self._get_status_and_id(annotation_endpoint)

        original_response = self._download_reviewing_data(id)

        return original_response
