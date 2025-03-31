import base64
import json
from io import BytesIO
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import requests

from edenai_apis.features import AudioInterface, ProviderInterface
from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
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
                    code=500, message="Internal Server Error"
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
