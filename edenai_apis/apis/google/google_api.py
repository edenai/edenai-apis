import os
from typing import Dict

from edenai_apis.apis.google.google_audio_api import GoogleAudioApi
from edenai_apis.apis.google.google_image_api import GoogleImageApi
from edenai_apis.apis.google.google_ocr_api import GoogleOcrApi
from edenai_apis.apis.google.google_text_api import GoogleTextApi
from edenai_apis.apis.google.google_translation_api import GoogleTranslationApi
from edenai_apis.apis.google.google_video_api import GoogleVideoApi
from edenai_apis.features import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from google.cloud import storage
from google.cloud import translate_v3 as translate
from google.cloud import videointelligence, vision
from google.cloud.language import LanguageServiceClient


class GoogleApi(
    ProviderInterface,
    GoogleTextApi,
    GoogleImageApi,
    GoogleOcrApi,
    GoogleTranslationApi,
    GoogleAudioApi,
    GoogleVideoApi,
):
    provider_name = "google"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings, self.location = load_provider(
            ProviderDataEnum.KEY,
            provider_name=self.provider_name,
            location=True,
            api_keys=api_keys,
        )
        self.project_id = self.api_settings["project_id"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.location

        self.clients = {
            "image": vision.ImageAnnotatorClient(),
            "text": LanguageServiceClient(),
            "storage": storage.Client(),
            "video": videointelligence.VideoIntelligenceServiceClient(),
            "translate": translate.TranslationServiceClient(),
        }
