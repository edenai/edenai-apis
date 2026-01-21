import base64
from io import BytesIO
from typing import Dict, Optional

import httpx
import requests

from edenai_apis.features import AudioInterface
from edenai_apis.utils.http_client import async_client, AUDIO_TIMEOUT
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.tts import TtsDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.tts import get_tts_config
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
    aupload_file_bytes_to_s3,
)

from .config import voice_ids, get_audio_format_and_extension, DEFAULT_OUTPUT_FORMAT


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
        return "eleven_multilingual_v2"

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

    async def audio__atext_to_speech(
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

        ids = ElevenlabsApi.__get_voice_id(voice_id=voice_id)
        url = f"{self.base_url}text-to-speech/{ids}"
        model = ElevenlabsApi.__get_model_from_voice(voice_id=voice_id)
        data = {
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }
        async with async_client(AUDIO_TIMEOUT) as client:
            response = await client.post(url, json=data, headers=self.headers)

            if response.status_code != 200:
                raise ProviderException(response.text, code=response.status_code)

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = await aupload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )

        return ResponseType[TextToSpeechDataClass](
            original_response=audio,
            standardized_response=TextToSpeechDataClass(
                audio=audio, voice_type=1, audio_resource_url=resource_url
            ),
        )

    async def audio__atts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = DEFAULT_OUTPUT_FORMAT,
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using ElevenLabs API (async version).

        Args:
            text: The text to convert to speech
            model: The model to use (e.g., "eleven_multilingual_v2", "eleven_v3",
                   "eleven_flash_v2_5", "eleven_turbo_v2_5").
                   Defaults to "eleven_multilingual_v2"
            voice: The voice ID or name (e.g., "Rachel", "21m00Tcm4TlvDq8ikWAM").
                   Defaults to "Rachel"
            audio_format: Audio format (mp3, wav, pcm, opus, alaw, ulaw).
                   Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0, mapped to 0.7-1.2). Defaults to 1.0
            provider_params: Additional ElevenLabs API parameters passed to voice_settings:
                - stability: Voice stability (0.0 to 1.0, default 0.5)
                - similarity_boost: Voice similarity (0.0 to 1.0, default 0.75)
                - style: Style exaggeration (0.0 to 1.0, default 0)
                - use_speaker_boost: Boost similarity (boolean, default True)
        """
        provider_params = provider_params or {}
        config = get_tts_config("elevenlabs")

        # Set defaults
        resolved_model = model or config["default_model"]
        resolved_voice = voice or config["default_voice"]

        # Resolve voice name to voice ID (case-insensitive lookup using lowercase)
        voice_lower = resolved_voice.lower()
        if voice_lower in config["voice_ids"]:
            voice_id = config["voice_ids"][voice_lower]
        else:
            # Assume it's already a voice ID
            voice_id = resolved_voice

        url = f"{self.base_url}text-to-speech/{voice_id}"

        # Map standard speed range (0.25-4.0) to ElevenLabs' range (0.7-1.2)
        mapped_speed = 1.0
        if speed is not None:
            if speed <= 1.0:
                # Map [0.25, 1.0] -> [0.7, 1.0]
                normalized = (speed - 0.25) / 0.75
                mapped_speed = 0.7 + normalized * 0.3
            else:
                # Map (1.0, 4.0] -> (1.0, 1.2]
                normalized = (speed - 1.0) / 3.0
                mapped_speed = 1.0 + normalized * 0.2

        # Convert audio format to ElevenLabs format and get file extension
        elevenlabs_format, file_extension = get_audio_format_and_extension(audio_format)

        # Voice settings with defaults, overridden by provider_params
        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": mapped_speed,
            **provider_params,
        }

        data = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": voice_settings,
            "output_format": elevenlabs_format,
        }

        try:
            async with async_client(AUDIO_TIMEOUT) as client:
                response = await client.post(url, json=data, headers=self.headers)
                response.raise_for_status()

            audio_content = BytesIO(response.content)
            resource_url = await aupload_file_bytes_to_s3(
                audio_content, f".{file_extension}", USER_PROCESS
            )

            return ResponseType[TtsDataClass](
                original_response={},
                standardized_response=TtsDataClass(audio_resource_url=resource_url),
            )
        except httpx.TimeoutException as exc:
            raise ProviderException(message="Request timed out", code=408) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderException(
                exc.response.text, code=exc.response.status_code
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderException(message=f"Request failed: {exc}", code=500) from exc

    def audio__tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = DEFAULT_OUTPUT_FORMAT,
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using ElevenLabs API (sync version).

        Args:
            text: The text to convert to speech
            model: The model to use (e.g., "eleven_multilingual_v2", "eleven_v3",
                   "eleven_flash_v2_5", "eleven_turbo_v2_5").
                   Defaults to "eleven_multilingual_v2"
            voice: The voice ID or name (e.g., "Rachel", "21m00Tcm4TlvDq8ikWAM").
                   Defaults to "Rachel"
            audio_format: Audio format (mp3, wav, pcm, opus, alaw, ulaw).
                   Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0, mapped to 0.7-1.2). Defaults to 1.0
            provider_params: Additional ElevenLabs API parameters passed to voice_settings:
                - stability: Voice stability (0.0 to 1.0, default 0.5)
                - similarity_boost: Voice similarity (0.0 to 1.0, default 0.75)
                - style: Style exaggeration (0.0 to 1.0, default 0)
                - use_speaker_boost: Boost similarity (boolean, default True)
        """
        provider_params = provider_params or {}
        config = get_tts_config("elevenlabs")

        # Set defaults
        resolved_model = model or config["default_model"]
        resolved_voice = voice or config["default_voice"]

        # Resolve voice name to voice ID (case-insensitive lookup using lowercase)
        voice_lower = resolved_voice.lower()
        if voice_lower in config["voice_ids"]:
            voice_id = config["voice_ids"][voice_lower]
        else:
            # Assume it's already a voice ID
            voice_id = resolved_voice

        url = f"{self.base_url}text-to-speech/{voice_id}"

        # Map standard speed range (0.25-4.0) to ElevenLabs' range (0.7-1.2)
        mapped_speed = 1.0
        if speed is not None:
            if speed <= 1.0:
                # Map [0.25, 1.0] -> [0.7, 1.0]
                normalized = (speed - 0.25) / 0.75
                mapped_speed = 0.7 + normalized * 0.3
            else:
                # Map (1.0, 4.0] -> (1.0, 1.2]
                normalized = (speed - 1.0) / 3.0
                mapped_speed = 1.0 + normalized * 0.2

        # Convert audio format to ElevenLabs format and get file extension
        elevenlabs_format, file_extension = get_audio_format_and_extension(audio_format)

        # Voice settings with defaults, overridden by provider_params
        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": mapped_speed,
            **provider_params,
        }

        data = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": voice_settings,
            "output_format": elevenlabs_format,
        }

        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise ProviderException(message="Request timed out", code=408) from exc
        except requests.exceptions.HTTPError as exc:
            raise ProviderException(response.text, code=response.status_code) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=f"Request failed: {exc}", code=500) from exc

        audio_content = BytesIO(response.content)
        resource_url = upload_file_bytes_to_s3(
            audio_content, f".{file_extension}", USER_PROCESS
        )

        return ResponseType[TtsDataClass](
            original_response={},
            standardized_response=TtsDataClass(audio_resource_url=resource_url),
        )
