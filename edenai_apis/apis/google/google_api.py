import os
from typing import Dict

from google.cloud import aiplatform, storage
from google.cloud import translate_v3 as translate
from google.cloud import videointelligence, vision
from edenai_apis.loaders.utils import check_empty_values
from google.cloud.language import LanguageServiceClient
from google.oauth2 import service_account

from edenai_apis.apis.google.google_audio_api import GoogleAudioApi
from edenai_apis.apis.google.google_image_api import GoogleImageApi
from edenai_apis.apis.google.google_llm_api import GoogleLLMApi
from edenai_apis.apis.google.google_ocr_api import GoogleOcrApi
from edenai_apis.apis.google.google_text_api import GoogleTextApi
from edenai_apis.apis.google.google_translation_api import GoogleTranslationApi
from edenai_apis.apis.google.google_video_api import GoogleVideoApi
from edenai_apis.apis.google.google_multimodal_api import GoogleMultimodalApi
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.llmengine.llm_engine import LLMEngine


class GoogleApi(
    ProviderInterface,
    GoogleTextApi,
    GoogleImageApi,
    GoogleOcrApi,
    GoogleTranslationApi,
    GoogleAudioApi,
    GoogleVideoApi,
    GoogleMultimodalApi,
    GoogleLLMApi,
):
    provider_name = "google"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings, self.location = load_provider(
            ProviderDataEnum.KEY,
            provider_name=self.provider_name,
            location=True,
            api_keys=api_keys,
        )
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings["webhook_token"]
        self.project_id = self.api_settings["project_id"]

        clients_init_payload = {}
        self.clients = {
            "llm_client": LLMEngine(
                provider_name="gemini",
                provider_config={
                    "api_key": self.api_settings.get("genai_api_key"),
                },
            ),
        }
        if self.location:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.location
            self.__set_remaining_clients(clients_init_payload)
        else:
            if not check_empty_values(self.api_settings, ["genai_api_key"]):
                credentials = service_account.Credentials.from_service_account_info(
                    self.api_settings
                )
                clients_init_payload["credentials"] = credentials
                self.__set_remaining_clients(clients_init_payload)

        aiplatform.init(project=self.project_id, **clients_init_payload)

    def __set_remaining_clients(self, payload: Dict):
        self.clients = {
            **self.clients,
            "image": vision.ImageAnnotatorClient(**payload),
            "text": LanguageServiceClient(**payload),
            "storage": storage.Client(**payload),
            "video": videointelligence.VideoIntelligenceServiceClient(**payload),
            "translate": translate.TranslationServiceClient(**payload),
        }
