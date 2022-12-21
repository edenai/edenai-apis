
from io import BufferedReader
import requests
import json
from edenai_apis.features import ProviderApi, Audio
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import file_with_good_extension
from apis.amazon.helpers import check_webhook_result


class DeepgramApi(ProviderApi, Audio):
    provider_name = "deepgram"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["deepgram_key"]
        self.url = self.api_settings["url"]
        self.webhook_token = self.api_settings["webhook_token"]
        self.webhook_url = f"https://webhook.site/{self.webhook_token}"



    def audio__speech_to_text_async__launch_job(self, file: BufferedReader, 
        language: str, speakers: int, profanity_filter: bool, vocabulary: list
        ) -> AsyncLaunchJobResponseType:

        # check if audio file needs convertion
        accepted_extensions = ["wav", "mp2", "mp3", "mp4", "flac","aac","pcm","m4m", "ogg", "opus", "webm"]
        new_file, export_format, channels, frame_rate = file_with_good_extension(file, accepted_extensions)

        headers = {
            "authorization" : f"Token {self.api_key}",
            "content-type" : f"audio/{export_format}"
        }

        data_config = {
            "language" : language,
            "callback" : self.webhook_url,
            "punctuate" : "true",
            "diarize": "true",
            "multichannel" : "true"
        }
        if not language:
            del data_config["language"]
            data_config.update({
                "detect_language" : True
            })
        for key,value in data_config.items():
            self.url = (
                f"{self.url}&{key}={value}"
                if "?" in self.url
                else f"{self.url}?{key}={value}"
            )

        response = requests.post(self.url, headers=headers, json=data_config, files= {"files": new_file})
        result = response.json()
        if response.status_code != 200:
            print(result)
            raise ProviderException(f"{result.get('err_code')}: {result.get('err_msg')}")

        transcribe_id = response.json()["request_id"]
        return AsyncLaunchJobResponseType(
            provider_job_id= transcribe_id
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        
        # Getting results from webhook.site
        original_response, response_status = check_webhook_result(provider_job_id, self.api_settings)
        if original_response is None :
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        if response_status != 200:
            raise ProviderException(original_response)
        
        text = ""
        diarization_entries = []
        speakers = set()
        
        if original_response.get("err_code"):
            raise ProviderException(f"{original_response.get('err_code')}: {original_response.get('err_msg')}")

        channels = original_response["results"].get("channels", [])
        for channel in channels:
            text_response = channel["alternatives"][0]
            text = text + text_response["transcript"]
            for word in text_response.get("words", []):
                speaker = word.get("speaker",0) + 1
                speakers.add(speaker)
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment=word["word"],
                        speaker=speaker,
                        start_time=str(word["start"]),
                        end_time= str(word["end"]),
                        confidence=word["confidence"]
                    )
                )

        diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)
        standardized_response = SpeechToTextAsyncDataClass(text=text.strip(), diarization=diarization)
        return AsyncResponseType(
            original_response=original_response,
            standardized_response= standardized_response,
            provider_job_id= provider_job_id
        )
