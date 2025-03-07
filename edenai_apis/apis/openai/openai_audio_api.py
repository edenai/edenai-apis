import base64
import json
import uuid
from io import BytesIO
from typing import List, Optional

import requests

from edenai_apis.apis.amazon.helpers import check_webhook_result
from edenai_apis.features import AudioInterface
from edenai_apis.features.audio import TextToSpeechDataClass
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3

from .helpers import convert_tts_audio_rate


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
