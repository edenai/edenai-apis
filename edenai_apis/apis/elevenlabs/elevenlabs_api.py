import base64
from io import BytesIO
from typing import Dict

import requests

from edenai_apis.features import AudioInterface
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .config import voice_ids


class ElevenlabsApi(ProviderInterface, AudioInterface):
    provider_name = "elevenlabs"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.own_keys = bool(api_keys)
        self.base_url = "https://api.elevenlabs.io/v1/"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

    def __get_model_from_voice(voice_id: str):
        if "Multilingual" in voice_id:
            return "eleven_multilingual_v2"
        return "eleven_monolingual_v1"

    def __get_voice_id(voice_id: str):
        try:
            voice_name = voice_id.split("_")[-1]  # Extract the name from the voice_id
            voice_id_from_dict = voice_ids[
                voice_name
            ]  # Retrieve the ID using the name from the dict
        except Exception:
            raise ProviderException("Voice ID not found for the given voice name.")
        return voice_id_from_dict

    def __moderate_content(self, text: str):
        api_settings = load_provider(ProviderDataEnum.KEY, "openai")
        api_key = api_settings["api_key"]
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.post(
            "https://api.openai.com/v1/moderations",
            headers=headers,
            json={"input": text},
        )
        try:
            response_data = response.json()
            if "error" in response_data or response.status_code >= 400:
                return False
            flagged = response_data["results"][0]["flagged"]
        except Exception:
            return False
        if flagged:
            raise ProviderException(
                message="Content rejected due to violation of content policies.",
                code=400,
            )
        return False

    def audio__text_to_speech(
        self,
        language: str,
        text: str,
        option: str,
        voice_id: str,
        audio_format: str,
        speaking_rate: int,
        speaking_pitch: int,
        speaking_volume: int,
        sampling_rate: int,
        **kwargs,
    ) -> ResponseType[TextToSpeechDataClass]:
        if not self.own_keys:
            self.__moderate_content(text=text)

        ids = ElevenlabsApi.__get_voice_id(voice_id=voice_id)
        url = f"{self.base_url}text-to-speech/{ids}"
        model = ElevenlabsApi.__get_model_from_voice(voice_id=voice_id)
        data = {
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }
        response = requests.post(url, json=data, headers=self.headers)

        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )

        return ResponseType[TextToSpeechDataClass](
            original_response=audio,
            standardized_response=TextToSpeechDataClass(
                audio=audio, voice_type=1, audio_resource_url=resource_url
            ),
        )
