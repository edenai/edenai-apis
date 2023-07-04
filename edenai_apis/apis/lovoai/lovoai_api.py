import base64
from io import BytesIO
import json
from typing import Dict, Literal
import requests
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.audio import AudioInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.audio import retreive_voice_id
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


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
                    response.json().get("error", "Something went wrong")
                )
            except json.JSONDecodeError:
                raise ProviderException("Internal Server Error")

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
