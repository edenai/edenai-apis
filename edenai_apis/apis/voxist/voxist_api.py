from io import BufferedReader
import json
from typing import Dict
import requests
from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.features import Audio
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    AsyncLaunchJobResponseType
)


class VoxistApi(ProviderApi, Audio):
    provider_name: str = "voxist"

    def __init__(self) -> None:
        self.api_settings: Dict = load_provider(
            ProviderDataEnum.KEY, self.provider_name
        )
        self.username: str = self.api_settings["username"]
        self.password: str = self.api_settings["password"]
        self.base_url: str = self.api_settings["base_url"]
        self._connection()

    def _connection(self) -> None:
        """Methods to connect to the API"""
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }

        response = requests.post(f"{self.base_url}oauth/token", json=data)
        self.api_key = response.json().get("access_token")

    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str
    ) -> AsyncLaunchJobResponseType:
        # Convert audio file to Mono 16kHz wav
        wav_file = wav_converter(file, frame_rate=16000, channels=1)[0]

        # Prepare data
        headers = {"Authorization": f"Bearer {self.api_key}"}

        config = {
            "punctuation": "True",
            "diarization": "True",
            "lang": language,
            "sample_rate": 16000,
        }

        data = {"config": json.dumps(config)}

        files = [("file_channel1", wav_file)]

        # Call Api
        response = requests.post(
            url=f"{self.base_url}transcription", headers=headers, files=files, data=data
        )

        if response.status_code == 504:
            raise ProviderException(message="Gateway Timeout")

        original_response = response.json()

        # Raise error
        if response.status_code != 200:
            raise ProviderException(message=original_response.get("error"))

        return AsyncLaunchJobResponseType(
            provider_job_id=original_response.get("jobid")
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(
            url=f"{self.base_url}jobs/{provider_job_id}", headers=headers
        )

        # Code HTTP 202 in the next version of voxist's api
        if response.status_code == 202:
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )

        if response.status_code != 200:
            error = response.json().get("error")
            raise ProviderException(error)


        original_response = response.json()
        text = ""

        for i, word in enumerate(original_response):
            text += word["Lexical"]
            if i != len(original_response) - 1:
                text += " "

        standarized_response = SpeechToTextAsyncDataClass(text=text)

        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
            provider_job_id=provider_job_id,
        )
