from io import BufferedReader
import json
import re
from typing import Dict, List, Optional
import requests
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import AsyncJobException, AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.files import FileWrapper
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    AsyncLaunchJobResponseType
)


class VoxistApi(ProviderInterface, AudioInterface):
    provider_name: str = "voxist"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings: Dict = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys = api_keys
        )
        self.username: str = self.api_settings["username"]
        self.password: str = self.api_settings["password"]
        self.base_url: str = "https://asr-lvl.voxist.com/"
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
        self, 
        file: str, 
        language: str, 
        speakers: int,
        profanity_filter: bool, 
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model : str = None,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes

        # Prepare data
        headers = {"Authorization": f"Bearer {self.api_key}"}

        config = {
            "diarization": "True",
            "lang": language,
            "sample_rate": 16000,
        }

        data = {"config": json.dumps(config)}

        file_ = open(file, "rb")
        files = [("file_channel1", file_)]

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
        
        if error := response.json().get("error"):
            if re.match(r"jobid (.)* is invalid", error):
                raise AsyncJobException(
                    reason = AsyncJobExceptionReason.DEPRECATED_JOB_ID
                )
            raise ProviderException(error)

        diarization_entries = []
        speakers = set()

        original_response = response.json()
        text = ""
        print(original_response)

        for i, phrase in enumerate(original_response):
            text += phrase["Lexical"]
            if i != len(original_response) - 1:
                text += " "
            
            speakers.add(phrase['Speaker'])
            for word in phrase['Words']:
                start_time = phrase['Start_time']+ word['Offset']
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment= word['Word'],
                        speaker= int(phrase['Speaker'].split('_')[-1]) + 1,
                        start_time= str(start_time),
                        end_time= str(start_time+ word['Duration']),
                        confidence= word['Confidence']
                    )
                )
        
        diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)

        standardized_response = SpeechToTextAsyncDataClass(text=text, diarization=diarization)

        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )
