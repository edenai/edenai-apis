import random
from typing import Dict
import asyncio
import aiohttp

import openai
from openai import OpenAI

from edenai_apis.apis.openai.openai_doc_parsing_api import OpenaiDocParsingApi
from edenai_apis.apis.openai.openai_audio_api import OpenaiAudioApi
from edenai_apis.apis.openai.openai_image_api import OpenaiImageApi
from edenai_apis.apis.openai.openai_text_api import OpenaiTextApi
from edenai_apis.apis.openai.openai_translation_api import OpenaiTranslationApi
from edenai_apis.apis.openai.openai_multimodal_api import OpenaiMultimodalApi
from edenai_apis.apis.openai.helpers import moderate_if_exists
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from asgiref.sync import async_to_sync


class OpenaiApi(
    ProviderInterface,
    OpenaiImageApi,
    OpenaiTranslationApi,
    OpenaiTextApi,
    OpenaiAudioApi,
    OpenaiMultimodalApi,
    OpenaiDocParsingApi,
):
    provider_name = "openai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )

        if isinstance(self.api_settings, list):
            chosen_api_setting = random.choice(self.api_settings)
        else:
            chosen_api_setting = self.api_settings

        self.api_key = chosen_api_setting["api_key"]
        openai.api_key = self.api_key
        self.org_key = chosen_api_setting["org_key"]
        self.url = "https://api.openai.com/v1"
        self.model = "gpt-3.5-turbo-instruct"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
            "Content-Type": "application/json",
        }
        self.max_tokens = 270

        self.client = OpenAI(
            api_key=self.api_key,
        )

        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings["webhook_token"]
        self.moderation_flag = True

    async def _check_content_moderation(self, *args, **kwargs):
        tasks = []

        tasks.append(moderate_if_exists(self.headers, kwargs.get("text")))
        tasks.append(
            moderate_if_exists(self.headers, kwargs.get("chatbot_global_action"))
        )
        tasks.append(moderate_if_exists(self.headers, kwargs.get("instruction")))

        if "previous_history" in kwargs:
            tasks.extend(
                moderate_if_exists(self.headers, item.get("message"))
                for item in kwargs["previous_history"]
                if isinstance(item, dict)
            )

        if "texts" in kwargs:
            tasks.extend(
                moderate_if_exists(self.headers, item) for item in kwargs["texts"]
            )

        if "messages" in kwargs:
            for message in kwargs["messages"]:
                if isinstance(message, dict) and "content" in message:
                    for content in message["content"]:
                        if isinstance(content, dict) and "content" in content:
                            tasks.append(
                                moderate_if_exists(
                                    self.headers, content["content"].get("text")
                                )
                            )

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*tasks)

    def check_content_moderation(self, *args, **kwargs):
        async_to_sync(self._check_content_moderation)(*args, **kwargs)
