"""
Fake provider class used for tests
"""

from io import BufferedReader
from random import randint
from time import sleep
from typing import Dict, List, Optional

from edenai_apis.features import AudioInterface, ProviderInterface
from edenai_apis.features.audio import SpeechToTextAsyncDataClass, SpeechDiarization
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)


class FakerApi(ProviderInterface, AudioInterface):
    provider_name = "faker"

    def __init__(self, api_keys: Dict = {}) -> None:
        super().__init__()

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
        sleep(randint(1, 3))
        return AsyncLaunchJobResponseType(provider_job_id="SomeFakeID")

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        sleep(randint(1, 3))

        standardized_response = SpeechToTextAsyncDataClass(
            text="empty", diarization=SpeechDiarization(total_speakers=1)
        )
        provider_correct_response = AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response={},
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )

        chance_to_stop = randint(2, 1000)

        if provider_job_id == "FINISHED":
            return provider_correct_response
        if provider_job_id == "ERROR":
            raise Exception("error")
        if provider_job_id == "pending":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        if chance_to_stop < 250:
            return provider_correct_response
        if chance_to_stop > 994:
            raise Exception("error")
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
