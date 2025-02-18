from pathlib import Path
from time import time
from typing import Dict, List, Optional

import requests

from edenai_apis.features import ProviderInterface, AudioInterface
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.upload_s3 import upload_file_to_s3
from .helper import language_matches


class AssemblyApi(ProviderInterface, AudioInterface):
    provider_name = "assembly"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["assembly_key"]
        self.url = "https://api.assemblyai.com/v2"
        self.url_upload_file = f"{self.url}/upload"
        self.url_transcription = f"{self.url}/transcript"

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

        if language and "-" in language:
            language = language_matches[language]

        # upload file to server
        header = {"authorization": self.api_key}
        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )
        data = {
            "audio_url": f"{content_url}",
            "language_code": language,
            "speaker_labels": True,
            "filter_profanity": profanity_filter,
        }
        if vocabulary:
            data.update({"word_boost": vocabulary})
        if not language:
            del data["language_code"]
            data.update({"language_detection": True})
        data.update(provider_params)

        # if an option is not available for a language like 'speaker_labels' with french, we remove it
        launch_transcription = False
        trials = 10
        while not launch_transcription:
            trials -= 1
            # launch transcription
            response = requests.post(self.url_transcription, json=data, headers=header)
            if response.status_code != 200:
                error = response.json().get("error")
                if "not available in this language" in error:
                    parameter = error.split(":")[1].strip()
                    del data[parameter]
                else:
                    raise ProviderException(
                        response.json().get("error"), code=response.status_code
                    )
            else:
                launch_transcription = True

        transcribe_id = response.json()["id"]
        return AsyncLaunchJobResponseType(provider_job_id=transcribe_id)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        headers = {"authorization": self.api_key}

        response = requests.get(
            url=f"{self.url_transcription}/{provider_job_id}", headers=headers
        )

        if response.status_code != 200:
            error_message = (
                response.json().get("error") or "Error when transcribing audio file"
            )
            if "transcript id not found" in error_message:
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                    code=response.status_code,
                )
            raise ProviderException(error_message, code=response.status_code)

        diarization_entries = []
        speakers = {}
        index_speaker = 0

        original_response = response.json()
        status = original_response["status"]
        if status == "error":
            raise ProviderException(original_response, code=response.status_code)
        if status != "completed":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )

        # diarization
        if (
            original_response.get("utterances")
            and len(original_response["utterances"]) > 0
        ):
            for line in original_response["utterances"]:
                words = line.get("words", [])
                if line["speaker"] not in speakers:
                    index_speaker += 1
                    speaker_tag = index_speaker
                    speakers[line["speaker"]] = index_speaker
                elif line["speaker"] in speakers:
                    speaker_tag = speakers[line["speaker"]]
                for word in words:
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            speaker=speaker_tag,
                            segment=word["text"],
                            start_time=str(word["start"] / 1000),
                            end_time=str(word["end"] / 1000),
                            confidence=word["confidence"],
                        )
                    )

        diarization = SpeechDiarization(
            total_speakers=len(speakers), entries=diarization_entries
        )
        if len(speakers) == 0:
            diarization.error_message = (
                "Speaker diarization not available for the data specified"
            )
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=SpeechToTextAsyncDataClass(
                text=original_response["text"], diarization=diarization
            ),
            provider_job_id=provider_job_id,
        )
