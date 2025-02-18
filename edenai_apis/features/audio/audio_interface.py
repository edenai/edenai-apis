from abc import ABC, abstractmethod
from io import BufferedReader
from typing import List, Literal, Optional

from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.text_to_speech_async.text_to_speech_async_dataclass import (
    TextToSpeechAsyncDataClass,
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    ResponseType,
)


class AudioInterface:
    ### Speech to Text methods
    @abstractmethod
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
        """Launch an asynchronous job to convert an audio file to text
        Args:
            file (BufferedReader): audio file to analyze
            language (str): language code in ISO format
            speakers (int): number of speakers present in the audio
            profanity_filter (bool): whether or not to filter profanity and replace inappropriate words.
            vocabulary (list[str]): List of words or composed words to be detected by the speech to text engine
            provider_params (dict): default to {}, provider specific parameters
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
        """Convert Text into audio speech
        Args:
            language (str): language in ISO format
            text (str): text to convert
            option (Literal["MALE", "FEMALE"]):
        """
        raise NotImplementedError

    @abstractmethod
    def audio__text_to_speech_async__launch_job(
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
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        """Convert text into longer audio speech
        Args:
            language (str): language in ISO format
            text (str): text to convert
            option (Literal["MALE", "FEMALE"]):
        """
        raise NotImplementedError

    @abstractmethod
    def audio__text_to_speech_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[TextToSpeechAsyncDataClass]:
        """Get the result of an asynchronous job by its ID
        Args:
            - provider_job_id (str): id of async job
        """
        raise NotImplementedError

    @abstractmethod
    def audio__text_to_speech_async__get_results_from_webhook(
        self, data: dict
    ) -> AsyncBaseResponseType[TextToSpeechAsyncDataClass]:
        """
        Get the result of an asynchrous job from webhook

        Args:
            - data (dict): result data given by the provider
            when calling the webhook
        """
        raise NotImplementedError
