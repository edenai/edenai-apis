import base64
import json
from io import BytesIO
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import httpx
import requests

from edenai_apis.features import AudioInterface, ProviderInterface
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.tts import TtsDataClass
from edenai_apis.utils.http_client import async_client, AUDIO_TIMEOUT
from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.tts import get_tts_config
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
    aupload_file_bytes_to_s3,
    upload_file_to_s3,
)


class DeepgramApi(ProviderInterface, AudioInterface):
    provider_name = "deepgram"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["deepgram_key"]
        self.url = "https://api.deepgram.com/v1/listen"

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
        provider_params = provider_params or {}
        export_format, channels, frame_rate = audio_attributes

        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        # TODO handle local files: https://developers.deepgram.com/reference/listen-file

        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )

        headers = {
            "authorization": f"Token {self.api_key}",
            "content-type": f"application/json",
        }

        data = {"url": content_url}

        data_config = {
            "language": language,
            "punctuate": "true",
            "diarize": "true",
            "profanity_filter": "false",
            "tier": model,
        }
        if profanity_filter:
            data_config.update({"profanity_filter": "true"})

        if not language:
            del data_config["language"]
            data_config.update({"detect_language": "true"})

        data_config.update(provider_params)
        # deepgram doesn't accept boolean with python like syntax
        # as requests doesn't change the value (eg ?bool=True instead of ?bool=true)
        for key, value in data_config.items():
            if isinstance(value, bool):
                data_config[key] = str(value).lower()

        response = requests.post(
            self.url, headers=headers, json=data, params=data_config
        )
        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(
                f"{original_response.get('err_code')}: {original_response.get('err_msg')}",
                code=response.status_code,
            )

        text = ""
        diarization_entries = []
        res_speakers = set()

        if original_response.get("err_code"):
            raise ProviderException(
                f"{original_response.get('err_code')}: {original_response.get('err_msg')}",
                code=response.status_code,
            )

        channels = original_response["results"].get("channels", [])
        for channel in channels:
            text_response = channel["alternatives"][0]
            text = text + text_response["transcript"]
            for word in text_response.get("words", []):
                speaker = word.get("speaker", 0) + 1
                res_speakers.add(speaker)
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment=word["word"],
                        speaker=speaker,
                        start_time=str(word["start"]),
                        end_time=str(word["end"]),
                        confidence=word["confidence"],
                    )
                )

        diarization = SpeechDiarization(
            total_speakers=len(res_speakers), entries=diarization_entries
        )
        if profanity_filter:
            diarization.error_message = (
                "Profanity Filter converts profanity to the nearest "
                "recognized non-profane word or removes it from the transcript completely"
            )
        standardized_response = SpeechToTextAsyncDataClass(
            text=text.strip(), diarization=diarization
        )
        return AsyncResponseType(
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=original_response["metadata"]["request_id"],
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
        _, model = voice_id.split("_")
        base_url = f"https://api.deepgram.com/v1/speak?model={model}"
        if audio_format:
            base_url += f"&container={audio_format}"

        if sampling_rate:
            base_url += f"&sample_rate={sampling_rate}"

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}
        response = requests.post(
            base_url,
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            try:
                result = response.json()
            except json.JSONDecodeError as exc:
                raise ProviderException(
                    code=response.status_code, message=response.text
                ) from exc

            raise ProviderException(
                code=response.status_code, message=result.get("err_msg")
            )

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )

        return ResponseType[TextToSpeechDataClass](
            original_response=response.content,
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
        _, model = voice_id.split("_")
        base_url = f"https://api.deepgram.com/v1/speak?model={model}"
        if audio_format:
            base_url += f"&container={audio_format}"

        if sampling_rate:
            base_url += f"&sample_rate={sampling_rate}"

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}
        async with async_client(AUDIO_TIMEOUT) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            if response.status_code != 200:
                try:
                    result = response.json()
                except json.JSONDecodeError as exc:
                    raise ProviderException(
                        code=500, message="Internal Server Error"
                    ) from exc

                raise ProviderException(
                    code=response.status_code, message=result.get("err_msg")
                )
        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        audio_content.seek(0)
        resource_url = await aupload_file_bytes_to_s3(
            audio_content, f".{audio_format}", USER_PROCESS
        )

        return ResponseType[TextToSpeechDataClass](
            original_response=response.content,
            standardized_response=TextToSpeechDataClass(
                audio=audio, voice_type=1, audio_resource_url=resource_url
            ),
        )

    async def audio__atts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using Deepgram Aura API (async version).

        Args:
            text: The text to convert to speech
            model: The Aura model (e.g., "aura", "aura-2") or full voice ID
                   (e.g., "aura-asteria-en", "aura-2-thalia-en").
                   Defaults to "aura-asteria-en"
            voice: Alternative to model parameter (same functionality)
            audio_format: Audio format (mp3, wav, ogg). Defaults to "mp3"
            speed: Not supported by Deepgram (ignored)
            speaking_pitch: Not supported by Deepgram (ignored)
            speaking_volume: Not supported by Deepgram (ignored)
            provider_params: Additional Deepgram API parameters passed as query params:
                - sample_rate: Audio sample rate in Hz
                - encoding: Audio encoding (linear16, mulaw, alaw)
                - container: Audio container format
        """
        provider_params = provider_params or {}
        config = get_tts_config("deepgram")

        # Resolve voice - handles both model names ("aura", "aura-2") and full voice IDs
        resolved_model = self._resolve_deepgram_voice(model, voice, config)

        base_url = "https://api.deepgram.com/v1/speak"

        # Build params dict with model
        params = {"model": resolved_model}

        # Handle audio format - Deepgram requires encoding+container for non-mp3
        file_extension = audio_format
        if audio_format == "wav":
            params["encoding"] = "linear16"
            params["container"] = "wav"
        elif audio_format == "ogg":
            params["encoding"] = "opus"
            params["container"] = "ogg"
        # For mp3 (default), don't pass container - Deepgram defaults to mp3

        # Merge provider_params (allows overriding encoding/container if needed)
        params.update(provider_params)

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}

        try:
            async with async_client(AUDIO_TIMEOUT) as client:
                response = await client.post(base_url, headers=headers, json=payload, params=params)
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

    # Default voices for each model family (first English voice)
    MODEL_DEFAULT_VOICES = {
        "aura": "aura-asteria-en",
        "aura-2": "aura-2-thalia-en",
    }

    def _resolve_deepgram_voice(self, model: Optional[str], voice: Optional[str], config: dict) -> str:
        """Resolve the voice ID for Deepgram TTS.

        If only a model name is provided (e.g., "aura" or "aura-2"),
        returns the default English voice for that model.
        If voice is provided, validates it against model if specified.
        """
        model_lower = model.lower() if model else None
        voice_lower = voice.lower() if voice else None

        # If voice is provided, validate against model and return
        if voice_lower:
            # Validate voice matches model if both are specified
            if model_lower in self.MODEL_DEFAULT_VOICES:
                is_aura2_voice = voice_lower.startswith("aura-2-")
                if model_lower == "aura" and is_aura2_voice:
                    raise ProviderException(
                        f"Voice '{voice}' is not compatible with model '{model}'. "
                        f"Use an Aura-1 voice (e.g., 'aura-asteria-en') or model 'aura-2'.",
                        code=400
                    )
                if model_lower == "aura-2" and not is_aura2_voice:
                    raise ProviderException(
                        f"Voice '{voice}' is not compatible with model '{model}'. "
                        f"Use an Aura-2 voice (e.g., 'aura-2-thalia-en') or model 'aura'.",
                        code=400
                    )
            return voice_lower

        # If only model name provided, use default voice for that model
        if model_lower in self.MODEL_DEFAULT_VOICES:
            return self.MODEL_DEFAULT_VOICES[model_lower]

        # If model is a full voice ID, use it
        if model_lower:
            return model_lower

        # Fall back to config default
        return config["default_voice"].lower()

    def audio__tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using Deepgram Aura API (sync version).

        Args:
            text: The text to convert to speech
            model: The Aura model (e.g., "aura", "aura-2") or full voice ID
                   (e.g., "aura-asteria-en", "aura-2-thalia-en").
                   Defaults to "aura-asteria-en"
            voice: Alternative to model parameter (same functionality)
            audio_format: Audio format (mp3, wav, ogg). Defaults to "mp3"
            speed: Not supported by Deepgram (ignored)
            speaking_pitch: Not supported by Deepgram (ignored)
            speaking_volume: Not supported by Deepgram (ignored)
            provider_params: Additional Deepgram API parameters passed as query params:
                - sample_rate: Audio sample rate in Hz
                - encoding: Audio encoding (linear16, mulaw, alaw)
                - container: Audio container format
        """
        provider_params = provider_params or {}
        config = get_tts_config("deepgram")

        # Resolve voice - handles both model names ("aura", "aura-2") and full voice IDs
        resolved_model = self._resolve_deepgram_voice(model, voice, config)

        base_url = "https://api.deepgram.com/v1/speak"

        # Build params dict with model
        params = {"model": resolved_model}

        # Handle audio format - Deepgram requires encoding+container for non-mp3
        file_extension = audio_format
        if audio_format == "wav":
            params["encoding"] = "linear16"
            params["container"] = "wav"
        elif audio_format == "ogg":
            params["encoding"] = "opus"
            params["container"] = "ogg"
        # For mp3 (default), don't pass container - Deepgram defaults to mp3

        # Merge provider_params (allows overriding encoding/container if needed)
        params.update(provider_params)

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}

        try:
            response = requests.post(base_url, headers=headers, json=payload, params=params)
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise ProviderException(message="Request timed out", code=408) from exc
        except requests.exceptions.HTTPError as exc:
            try:
                result = response.json()
                message = result.get("err_msg", response.text)
            except json.JSONDecodeError:
                message = response.text
            raise ProviderException(code=response.status_code, message=message) from exc
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
