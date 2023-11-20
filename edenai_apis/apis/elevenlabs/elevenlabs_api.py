import base64
from io import BytesIO
from typing import Dict

import requests

from edenai_apis.features import AudioInterface
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import TextToSpeechDataClass
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
        self.base_url ="https://api.elevenlabs.io/v1/"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

    def __get_model_from_voice(voice_id: str):
        if 'Multilingual' in voice_id:
            return 'eleven_multilingual_v1'
        return 'eleven_monolingual_v1'
    
    def __get_voice_id(voice_id: str):
        try:
            voice_name = voice_id.split('_')[-1]  # Extract the name from the voice_id
            voice_id_from_dict = voice_ids[voice_name]  # Retrieve the ID using the name from the dict
        except Exception:
            raise ProviderException("Voice ID not found for the given voice name.")
        return voice_id_from_dict
    

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
        sampling_rate: int) -> ResponseType[TextToSpeechDataClass]:

        ids = ElevenlabsApi.__get_voice_id(voice_id=voice_id)
        url = f"{self.base_url}text-to-speech/{ids}"
        model = ElevenlabsApi.__get_model_from_voice(voice_id=voice_id)
        data = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        response = requests.post(url, json=data, headers=self.headers)
        
        if response.status_code != 200:
            raise ProviderException(
                response.text,
                code = response.status_code
                )
        
        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, ".wav", USER_PROCESS)

        return ResponseType[TextToSpeechDataClass](
            original_response=audio,
            standardized_response=TextToSpeechDataClass(
                audio=audio, voice_type=1, audio_resource_url=resource_url
            ),
        )
