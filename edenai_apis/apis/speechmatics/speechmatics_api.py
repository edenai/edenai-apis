import json
from typing import Dict, Optional, List

import requests

from edenai_apis.features import ProviderInterface, AudioInterface
from edenai_apis.features.audio.speech_to_text_async import (
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
    AsyncResponseType,
    AsyncPendingResponseType,
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
)


class SpeechmaticsApi(ProviderInterface, AudioInterface):
    provider_name = "speechmatics"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.key = self.api_settings["speechmatics_key"]
        self.base_url = "https://asr.api.speechmatics.com/v2/jobs"
        self.headers = {
            "Authorization": f"Bearer {self.key}",
        }

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
        with open(file, "rb") as file_:
            config = {
                "language": language,
                "diarization": "speaker",
                "operating_point": model,
            }
            if vocabulary:
                config["additional_vocab"] = [{"content": word} for word in vocabulary]

            payload = {
                "config": json.dumps(
                    {"type": "transcription", "transcription_config": config}
                ),
                **provider_params,
            }
            # Send request
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                data=payload,
                files={"data_file": file_},
            )
        if response.status_code != 201:
            raise ProviderException(response.content, response.status_code)

        return AsyncLaunchJobResponseType(provider_job_id=response.json()["id"])

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        response = requests.get(
            f"{self.base_url}/{provider_job_id}", headers=self.headers
        )
        original_response = response.json()
        if response.status_code != 200:
            if original_response.get("details") == "path not found":
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                    code=response.status_code,
                )
            raise ProviderException(
                message=original_response,
                code=response.status_code,
            )

        job_details = original_response["job"]
        errors = job_details.get("errors")
        if errors:
            raise ProviderException(errors, code=response.status_code)

        status = job_details["status"]
        if status == "running":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        elif status == "done":
            response = requests.get(
                f"{self.base_url}/{provider_job_id}/transcript",
                headers=self.headers,
            )
            original_response = response.json()
            if response.status_code != 200:
                raise ProviderException(
                    original_response.get("errors"), code=response.status_code
                )

            diarization_entries = []
            speakers = set()
            text = ""
            for entry in original_response.get("results"):
                text = text + " " + entry["alternatives"][0]["content"]
                speaker = entry["alternatives"][0].get("speaker") or None
                if speaker:
                    speakers.add(speaker)

                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment=entry["alternatives"][0]["content"],
                        start_time=str(entry["start_time"]),
                        end_time=str(entry["end_time"]),
                        confidence=entry["alternatives"][0]["confidence"],
                        speaker=(list(speakers).index(speaker) + 1 if speaker else 0),
                    )
                )
            diarization = SpeechDiarization(
                total_speakers=len(speakers), entries=diarization_entries
            )
            return AsyncResponseType(
                original_response=original_response,
                standardized_response=SpeechToTextAsyncDataClass(
                    text=text, diarization=diarization
                ),
                provider_job_id=provider_job_id,
            )
        else:
            raise ProviderException("Unexpected job failed")
