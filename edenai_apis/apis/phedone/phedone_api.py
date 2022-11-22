import requests

from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
)
from edenai_apis.features import Translation
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class PhedoneApi(ProviderApi, Translation):
    """
    attributes:
      provider_name: str = 'phedone'
    """

    provider_name: str = "phedone"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.base_url = self.api_settings["base_url"]

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        """
        Parameters:
          source_languages: str
          target_languages: str
          text: str

        Return:
          {
            original_response: {},
            standarized_response: {},
          }
        """

        if not source_language:
            source_language = "auto"
        file = {
            "text": text,
            "input_locale": source_language,
            "output_locale": target_language,
        }
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}translation"

        response = requests.post(url=url, headers=headers, json=file)

        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(original_response.get("message"))

        standarized_response = AutomaticTranslationDataClass(
            text=original_response.get("translation")[0]
        )

        result = ResponseType[AutomaticTranslationDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

        return result
