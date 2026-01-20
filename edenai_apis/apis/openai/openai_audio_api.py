import base64
import uuid
from io import BytesIO
from typing import List, Optional

import httpx
import requests

from edenai_apis.features import AudioInterface
from edenai_apis.utils.http_client import async_client, AUDIO_TIMEOUT
from edenai_apis.features.audio import TextToSpeechDataClass
from edenai_apis.features.audio.tts import TtsDataClass
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
    aupload_file_bytes_to_s3,
)
from edenai_apis.utils.tts import normalize_speed_for_openai
from edenai_apis.loaders.data_loader import load_provider_subfeature_info

from .helpers import convert_tts_audio_rate


def _get_tts_constraints():
    """Load TTS constraints from info.json"""
    info = load_provider_subfeature_info("openai", "audio", "tts")
    return info.get("constraints", {})


class OpenaiAudioApi(AudioInterface):
    def _speech_to_text(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
    ):
        provider_params = provider_params or {}
        data_job_id = {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
        }
        url = "https://api.openai.com/v1/audio/transcriptions"
        with open(file, "rb") as file_:
            files = {"file": file_}
            payload = {"model": "whisper-1", "language": language, **provider_params}
            response = requests.post(url, data=payload, files=files, headers=headers)
            if response.status_code != 200:
                raise ProviderException(response.text, response.status_code)

        try:
            original_response = response.json()
        except requests.JSONDecodeError as exp:
            raise ProviderException("Internal Server Error", code=500) from exp
        diarization = SpeechDiarization(total_speakers=0, entries=[])
        standardized_response = SpeechToTextAsyncDataClass(
            text=original_response.get("text"), diarization=diarization
        )
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=str(uuid.uuid4()),
        )

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        return self._speech_to_text(
            file,
            language,
            speakers,
            profanity_filter,
            vocabulary,
            audio_attributes,
            model,
            file_url,
            provider_params,
        )

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
        url = "https://api.openai.com/v1/audio/speech"
        speed = convert_tts_audio_rate(speaking_rate)
        if not audio_format:
            audio_format = "mp3"
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice_id[3:],
            "speed": speed,
            "response_format": audio_format,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = response.content
        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1
        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )
        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url=resource_url
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={}, standardized_response=standardized_response
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
        url = "https://api.openai.com/v1/audio/speech"
        speed = convert_tts_audio_rate(speaking_rate)
        if not audio_format:
            audio_format = "mp3"
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice_id[3:],
            "speed": speed,
            "response_format": audio_format,
        }
        async with async_client(AUDIO_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=self.headers)

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1
        audio_content.seek(0)
        resource_url = await aupload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )
        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url=resource_url
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={}, standardized_response=standardized_response
        )

    async def audio__atts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using OpenAI's TTS API (async version).

        Args:
            text: The text to convert to speech
            model: The TTS model (e.g., "gpt-4o-mini-tts", "tts-1", "tts-1-hd").
                   Defaults to value from info.json
            voice: The voice ID (e.g., "alloy", "ash", "ballad", "coral", "echo",
                   "fable", "marin", "cedar", "nova", "onyx", "sage", "shimmer", "verse").
                   Defaults to value from info.json
            audio_format: Audio format (mp3, opus, aac, flac, wav, pcm). Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0). Defaults to 1.0
            provider_params: Additional provider-specific parameters (not used for OpenAI)
        """
        url = "https://api.openai.com/v1/audio/speech"

        # Load constraints and set defaults from config
        constraints = _get_tts_constraints()
        resolved_model = model or constraints.get("default_model", "tts-1")
        resolved_voice = voice or constraints.get("default_voice", "alloy")
        resolved_speed = normalize_speed_for_openai(speed)

        payload = {
            "model": resolved_model,
            "input": text,
            "voice": resolved_voice,
            "speed": resolved_speed,
            "response_format": audio_format or "mp3",
        }

        try:
            async with async_client(AUDIO_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()

            audio_content = BytesIO(response.content)
            audio = base64.b64encode(audio_content.read()).decode("utf-8")
            audio_content.seek(0)
            resource_url = await aupload_file_bytes_to_s3(
                audio_content, f".{audio_format or 'mp3'}", USER_PROCESS
            )
            standardized_response = TtsDataClass(
                audio=audio, audio_resource_url=resource_url
            )
            return ResponseType[TtsDataClass](
                original_response={}, standardized_response=standardized_response
            )
        except httpx.TimeoutException as exc:
            raise ProviderException(message="Request timed out", code=408) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderException(exc.response.text, code=exc.response.status_code) from exc
        except httpx.RequestError as exc:
            raise ProviderException(message=f"Request failed: {exc}", code=500) from exc

    def audio__tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using OpenAI's TTS API (sync version).

        Args:
            text: The text to convert to speech
            model: The TTS model (e.g., "gpt-4o-mini-tts", "tts-1", "tts-1-hd").
                   Defaults to value from info.json
            voice: The voice ID (e.g., "alloy", "ash", "ballad", "coral", "echo",
                   "fable", "marin", "cedar", "nova", "onyx", "sage", "shimmer", "verse").
                   Defaults to value from info.json
            audio_format: Audio format (mp3, opus, aac, flac, wav, pcm). Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0). Defaults to 1.0
            provider_params: Additional provider-specific parameters (not used for OpenAI)
        """
        url = "https://api.openai.com/v1/audio/speech"

        # Load constraints and set defaults from config
        constraints = _get_tts_constraints()
        resolved_model = model or constraints.get("default_model", "tts-1")
        resolved_voice = voice or constraints.get("default_voice", "alloy")
        resolved_speed = normalize_speed_for_openai(speed)

        payload = {
            "model": resolved_model,
            "input": text,
            "voice": resolved_voice,
            "speed": resolved_speed,
            "response_format": audio_format or "mp3",
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=AUDIO_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise ProviderException(message="Request timed out", code=408) from exc
        except requests.exceptions.HTTPError as exc:
            raise ProviderException(response.text, code=response.status_code) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderException(message=f"Request failed: {exc}", code=500) from exc

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(
            audio_content, f".{audio_format or 'mp3'}", USER_PROCESS
        )
        standardized_response = TtsDataClass(
            audio=audio, audio_resource_url=resource_url
        )
        return ResponseType[TtsDataClass](
            original_response={}, standardized_response=standardized_response
        )
