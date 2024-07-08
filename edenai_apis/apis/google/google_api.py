import os
from typing import Dict

from google.cloud import aiplatform, storage
from google.cloud import translate_v3 as translate
from google.cloud import videointelligence, vision
from google.cloud.language import LanguageServiceClient
from google.oauth2 import service_account

from edenai_apis.apis.google.google_audio_api import GoogleAudioApi
from edenai_apis.apis.google.google_image_api import GoogleImageApi
from edenai_apis.apis.google.google_multimodal_api import GoogleMultimodalApi
from edenai_apis.apis.google.google_ocr_api import GoogleOcrApi
from edenai_apis.apis.google.google_text_api import GoogleTextApi
from edenai_apis.apis.google.google_translation_api import GoogleTranslationApi
from edenai_apis.apis.google.google_video_api import GoogleVideoApi
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


class GoogleApi(
    ProviderInterface,
    GoogleTextApi,
    GoogleImageApi,
    GoogleOcrApi,
    GoogleTranslationApi,
    GoogleAudioApi,
    GoogleVideoApi,
    GoogleMultimodalApi,
):
    provider_name = "google"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings, _ = load_provider(
            ProviderDataEnum.KEY,
            provider_name=self.provider_name,
            location=True,
            api_keys=api_keys,
        )

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self.credentials = service_account.Credentials.from_service_account_info(
            self.api_settings,
            scopes=scopes
        )
        self.project_id = self.api_settings["project_id"]

        self.clients = {
            "image": vision.ImageAnnotatorClient(credentials=self.credentials),
            "text": LanguageServiceClient(credentials=self.credentials),
            "storage": storage.Client(credentials=self.credentials),
            "video": videointelligence.VideoIntelligenceServiceClient(
                credentials=self.credentials
            ),
            "translate": translate.TranslationServiceClient(
                credentials=self.credentials
            ),
        }

        aiplatform.init(
            project=self.project_id,
        )
