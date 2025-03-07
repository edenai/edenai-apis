from typing import Dict, List, Optional

import requests

from edenai_apis.features import AudioInterface
from edenai_apis.features.audio.speech_to_text_async import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
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


class VociApi(ProviderInterface, AudioInterface):
    provider_name = "voci"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.key = self.api_settings["voci_key"]

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

        data_config = {
            "output": "json",
            "token": self.key,
            "filetype": f"audio/{export_format}",
            "emotion": "true",
            "gender": "true",
            "diarize": "true",
        }
        if language:
            data_config.update({"model": f"{language.lower()}:callcenter"})
        else:
            data_config.update({"lid": "true"})
        # if vocabulary:
        #     data_config.update({
        #         "hints": json.dumps({
        #             "other" : vocabulary
        #         })
        #     })

        data_config.update(provider_params)
        with open(file, "rb") as file_:
            response = requests.post(
                url="https://vcloud.vocitec.com/transcribe",
                data=data_config,
                files=[("file", file_)],
            )
        if response.status_code != 200:
            raise ProviderException(
                f"Call to Voci failed.\nResponse Status: {response.status_code}.\n"
                + f"Response Content: {response.content}",
                code=response.status_code,
            )
        else:
            original_response = response.json()
            job_id = original_response["requestid"]
            return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        payload = {"token": self.key, "requestid": provider_job_id}
        response = requests.get(
            url="https://vcloud.vocitec.com/transcribe/result", params=payload
        )
        if response.status_code == 200:
            url = response.json()
            response_text = requests.get(url=url)

            if response_text.status_code != 200:
                raise ProviderException(
                    f"Call to Voci failed.\nResponse Status: {response.status_code}.\n"
                    + f"Response Content: {response.text}",
                    code=response.status_code,
                )

            original_response = response_text.json()

            diarization_entries = []
            speakers = set()

            text = ""
            for utterance in original_response.get("utterances"):
                for event in utterance.get("events"):
                    text += event.get("word") + " "

                    speakers.add(utterance["metadata"]["channel"])
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            segment=event["word"],
                            start_time=str(event["start"]),
                            end_time=str(event["end"]),
                            confidence=event["confidence"],
                            speaker=utterance["metadata"]["channel"] + 1,
                        )
                    )
            diarization = SpeechDiarization(
                total_speakers=len(speakers), entries=diarization_entries
            )
            standardized_response = SpeechToTextAsyncDataClass(
                text=text, diarization=diarization
            )
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )
        elif response.status_code == 202:
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        else:
            error_message = response.text
            if "request ID" in error_message and any(
                portion in error_message for portion in ["Invalid", "Malformatted"]
            ):
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                    code=response.status_code,
                )
            raise ProviderException(response.text)
