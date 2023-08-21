import base64
from io import BytesIO
import json
from typing import Dict
import requests
from edenai_apis.features.audio.text_to_speech_async.text_to_speech_async_dataclass import TextToSpeechAsyncDataClass
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.audio import AudioInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType, ResponseType, AsyncResponseType, AsyncPendingResponseType
from edenai_apis.utils.exception import ProviderException, AsyncJobException, AsyncJobExceptionReason
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .config import voice_ids

class LovoaiApi(ProviderInterface, AudioInterface):
    provider_name = "lovoai"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.url = "https://api.lovo.ai/"

        self.headers = {
            "apiKey": self.api_settings["api_key"],
            "Content-Type": "application/json",
        }

    def __adjust_speaking_rate(self, speaking_rate: int):
        # convert value from  interval[-100 , 0 , 100] to [0.5 , 1 , 1.5]
        if speaking_rate > 100:
            speaking_rate = 100
        if speaking_rate < -100:
            speaking_rate = -100
        speaking_rate = 10 * round(speaking_rate / 10)
        return speaking_rate / 200 + 1

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
    ) -> ResponseType[TextToSpeechDataClass]:
        data = json.dumps(
            {
                "text": text,
                "speaker_id": voice_id.split("_")[-1],
                "speed": self.__adjust_speaking_rate(speaking_rate),
            }
        )

        response = requests.post(
            f"{self.url}v1/conversion", headers=self.headers, data=data
        )

        if response.status_code != 200:
            try:
                raise ProviderException(
                    response.json().get("error", "Something went wrong"),
                    code = response.status_code
                )
            except json.JSONDecodeError:
                raise ProviderException("Internal Server Error", code = 500)

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, ".wav", USER_PROCESS)

        return ResponseType[TextToSpeechDataClass](
            original_response={},
            standardized_response=TextToSpeechDataClass(
                audio=audio, voice_type=1, audio_resource_url=resource_url
            ),
        )

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
        file_url: str = "") -> AsyncLaunchJobResponseType:
        url = "https://api.genny.lovo.ai/api/v1/tts"
        data = json.dumps(
            {
                "speaker":voice_ids[voice_id],
                "text": text,
                "speed": self.__adjust_speaking_rate(speaking_rate),
            }
        )
        response = requests.post(
            url, headers={"X-API-KEY": self.api_settings["api_key_async"], "Content-Type": "application/json",}, 
            data=data
        )
        original_response = response.json()
        if response.status_code > 201:
            raise ProviderException(original_response.get("error"), code = original_response.get("statusCode"))

        return AsyncLaunchJobResponseType(provider_job_id=original_response["id"])
    
    def audio__text_to_speech_async__get_job_result(
        self, 
        provider_job_id: str) -> AsyncBaseResponseType[TextToSpeechAsyncDataClass]:
        headers = {
            "X-API-KEY": self.api_settings["api_key_async"], 
            "Content-Type": "application/json",
        }
        url_status = f"https://api.genny.lovo.ai/api/v1/tts/{provider_job_id}"

        response_status = requests.get(url=url_status, headers=headers)
        original_response = response_status.json()
        
        if response_status.status_code == 422 :
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID
            )
        
        if response_status.status_code != 200 :
            if "param: jobId is not valid objectId" in original_response.get("error"):
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID
                )
            raise ProviderException(original_response.get("error"), code = original_response.get("statusCode"))
        
        if original_response.get("status") != "done":
            return AsyncPendingResponseType[TextToSpeechAsyncDataClass](
                provider_job_id=provider_job_id
            )
            
        audio_url = original_response["data"][0]["urls"][0]
        audio_content = base64.b64encode(requests.get(audio_url).content)
        audio_content_string = audio_content.decode('utf-8')

        return AsyncResponseType[TextToSpeechAsyncDataClass](
            original_response=original_response,
            standardized_response=TextToSpeechAsyncDataClass(audio=audio_content_string,
                                                             voice_type=1,
                                                             audio_resource_url=audio_url),
            provider_job_id=provider_job_id,
        )