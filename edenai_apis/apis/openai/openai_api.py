from typing import Dict

from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.apis.openai.openai_doc_parsing_api import OpenaiDocParsingApi
from edenai_apis.apis.openai.openai_audio_api import OpenaiAudioApi
from edenai_apis.apis.openai.openai_image_api import OpenaiImageApi
from edenai_apis.apis.openai.openai_text_api import OpenaiTextApi
from edenai_apis.apis.openai.openai_translation_api import OpenaiTranslationApi
from edenai_apis.apis.openai.openai_multimodal_api import OpenaiMultimodalApi
from edenai_apis.apis.openai.openai_llm_api import OpenaiLLMApi
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
import openai
from openai import OpenAI


class OpenaiApi(
    ProviderInterface,
    OpenaiImageApi,
    OpenaiTranslationApi,
    OpenaiTextApi,
    OpenaiAudioApi,
    OpenaiMultimodalApi,
    OpenaiDocParsingApi,
    OpenaiLLMApi,
):
    provider_name = "openai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        
        self.api_key = self.api_settings.get("api_key")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.api_key = self.api_settings["api_key"]
        openai.api_key = self.api_key
        self.org_key = self.api_settings["org_key"]
        self.url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
            "Content-Type": "application/json",
        }
        self.max_tokens = 270

        self.client = OpenAI(
            api_key=self.api_key,
        )

        self.llm_client = LLMEngine(
            provider_name=self.provider_name,
            provider_config={
                "api_key": self.api_key,
            },
        )

        self.moderation_flag = True
