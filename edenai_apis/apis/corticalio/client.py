import json
from typing import Dict

import requests as requests
from requests import Response

from edenai_apis.utils.exception import ProviderException


class CorticalClient:
    """
    REST client for the Cortical.io NLP API
    """

    def __init__(self, api_settings: Dict) -> None:
        self.api_settings = api_settings
        self.auth_headers = {"Authorization": self.api_settings.get("api_key")}
        self.base_url = self.api_settings.get("base_url").rstrip("/")

    def extract_keywords(self, text: str, language: str):
        """
        Extract keywords from the provided text.

        Args:
            text (str): text of interest, required
            language (str): language of the specified text, optional

        Returns:
            List of keywords with corresponding scores and other metrics
        """
        response = requests.post(
            url=f"{self.base_url}/keywords",
            headers=self.auth_headers,
            json={"text": json.dumps(text), "language": language},
        )
        return self._handle_response(response)

    def _handle_response(self, response: Response):
        """
        Extract response JSON body and raise ProviderException on a failed request
        """
        try:
            response.raise_for_status()
            return response.json()
        except Exception:
            raise ProviderException(response.text, code=response.status_code)
