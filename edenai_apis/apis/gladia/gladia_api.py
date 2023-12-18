from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import Dict

import requests

from edenai_apis.features import ProviderInterface, AudioInterface
from edenai_apis.features.audio import SpeechDiarizationEntry, SpeechDiarization
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import upload_file_to_s3


class GladiaApi(ProviderInterface, AudioInterface):
    provider_name = "gladia"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["gladia_key"]
        self.url = "https://api.gladia.io/v2/transcription/"

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: list,
        audio_attributes: tuple,
        model: str = None,
        file_url: str = "",
        provider_params: dict = dict(),
    ) -> AsyncLaunchJobResponseType:
        headers = {"x-gladia-key": self.api_key}
        export_format, channels, frame_rate = audio_attributes
        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )
        data = {
            "audio_url": content_url,
            "detect_language": True,
            "enable_code_switching": True,
            "custom_vocabulary": vocabulary,
            "diarization": True,
            "diarization_config": {"number_of_speakers": speakers},
        }
        if language:
            data.update({"detect_language": False, "language": language})
        data.update(provider_params)
        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code != 201:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(message="Internal Server Error", code=500)
        return AsyncLaunchJobResponseType(
            provider_job_id=original_response.get("id", "")
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        headers = {"x-gladia-key": self.api_key, "accept": "application/json"}
        response = requests.get(self.url + provider_job_id, headers=headers)
        print(response.text)
        if response.status_code != 200:
            raise ProviderException(message=response.text, code=response.status_code)
        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(message="Internal Server Error", code=500)
        if original_response.get("status") == "processing":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        result = original_response.get("result", {})
        transcription = result.get("transcription", {})
        text = transcription.get("full_transcript", " ")
        entries = []
        total_speakers = 0
        for utterance in transcription.get("utterances", []):
            for word in utterance.get("words", []):
                entries.append(
                    SpeechDiarizationEntry(
                        segment=word.get("word", " "),
                        start_time=str(word.get("start", 0)),
                        end_time=str(word.get("end", 0)),
                        speaker=utterance.get("speaker", 0),
                        confidence=word.get("confidence", 0),
                    )
                )
            total_speakers = max(utterance.get("speaker", 0), total_speakers)
        diarization = SpeechDiarization(total_speakers=total_speakers, entries=entries)
        if total_speakers == 0:
            diarization.error_message = (
                "Speaker diarization not available for the data specified"
            )
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=SpeechToTextAsyncDataClass(
                text=text, diarization=diarization
            ),
            provider_job_id=provider_job_id,
        )
