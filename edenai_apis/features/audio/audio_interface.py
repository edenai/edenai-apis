from abc import abstractmethod
from io import BufferedReader
from typing import Literal

from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType, ResponseType


class AudioInterface:

    ### Speech to Text methods
    @abstractmethod
    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str
    ) -> AsyncLaunchJobResponseType:
        """Launch an asynchronous job to convert an audio file to text
        Args:
            file (BufferedReader): audio file to analyze
            language (str): language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def audio__speech_to_text_async__get_results_from_webhook(
        self, data: dict
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        """
        Get the result of an asynchrous job from webhook

        Args:
            - data (dict): result data given by the provider
            when calling the webhook
        """
        raise NotImplementedError

    ### Text to Speech methods
    @abstractmethod
    def audio__text_to_speech(
        self, language: str, text: str, option: Literal["MALE", "FEMALE"]
    ) -> ResponseType[TextToSpeechDataClass]:
        """Convert Text into audio speech
        Args:
            language (str): language in ISO format
            text (str): text to convert
            option (Literal["MALE", "FEMALE"]):
        """
        raise NotImplementedError
