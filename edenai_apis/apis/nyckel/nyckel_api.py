from typing import Dict
import requests
import time
from edenai_apis.apis.nyckel.nyckel_custom_image_classification import (
    NyckelCustomImageClassificationApi,
)
from edenai_apis.apis.nyckel.nyckel_image_api import NyckelImageApi
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException


class NyckelApi(ProviderInterface, NyckelImageApi, NyckelCustomImageClassificationApi):
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
        raise ProviderException(error_message, code=response.status_code)
