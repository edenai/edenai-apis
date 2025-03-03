import json
from typing import Dict, Optional

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    SummarizeDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class WritesonicApi(ProviderInterface, TextInterface):
    provider_name = "writesonic"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": self.api_key,
        }

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[SummarizeDataClass]:
        url = f"https://api.writesonic.com/v2/business/content/summary?engine=premium&language={language}"
        payload = {
            "article_text": text,
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error", code=500) from exc

        if "detail" in original_response:
            raise ProviderException(
                original_response["detail"], code=response.status_code
            )

        standardized_response = SummarizeDataClass(
            result=original_response[0].get("summary", {})
        )

        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    # def text__chat(
    #     self,
    #     text: str,
    #     chatbot_global_action: str = None,
    #     previous_history: List[Dict[str, str]] = None,
    #     temperature: float = 0,
    #     max_tokens: int = 2048,
    #     model: str = None) -> ResponseType[ChatDataClass]:

    #     url = "https://api.writesonic.com/v2/business/content/chatsonic?engine=premium"
    #     prompt = text
    #     if chatbot_global_action:
    #         prompt = chatbot_global_action + prompt
    #     payload = {
    #         "enable_google_results": "true",
    #         "enable_memory": False,
    #         "input_text": prompt
    #     }

    #     if previous_history:
    #         payload["enable_memory"] = True
    #         payload["history_data"] = []
    #         for message in previous_history:
    #             payload["history_data"].append(
    #                 {
    #                 "is_sent": "true",
    #                 "message": message.get("user")
    #                 }
    #             )
    #             payload["history_data"].append(
    #                 {
    #                 "is_sent": "false",
    #                 "message": message.get("assistant")
    #                 }
    #             )

    #     try:
    #         original_response = requests.post(url, json=payload, headers= self.headers).json()
    #     except json.JSONDecodeError as exc:
    #         raise ProviderException("Internal Server Error") from exc

    #     if "detail" in original_response:
    #         raise ProviderException(original_response['detail'])

    #     return ResponseType[ChatDataClass](
    #         original_response=original_response,
    #         standardized_response=ChatDataClass()
    #     )
