
from io import BufferedReader
import requests
import json
from time import time
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

from apis.amazon.config import storage_clients


class DeepgramApi(ProviderApi, Audio):
    provider_name = "deepgram"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_settings_amazon = load_provider(ProviderDataEnum.KEY, "amazon")
        self.api_key = self.api_settings["deepgram_key"]
        self.url = self.api_settings["url"]
        self.webhook_token = self.api_settings["webhook_token"]
        self.webhook_url = f"https://webhook.site/{self.webhook_token}"

        self.bucket_name = self.api_settings["bucket"]
        self.storage_url = self.api_settings["storage_url"]



    def audio__speech_to_text_async__launch_job(self, file: BufferedReader, 
        language: str, speakers: int, profanity_filter: bool, vocabulary: list
        ) -> AsyncLaunchJobResponseType:

        # check if audio file needs convertion
        accepted_extensions = ["wav", "mp2", "mp3", "mp4", "flac","aac","pcm","m4m", "ogg", "opus", "webm"]
        new_file, export_format, channels, frame_rate = file_with_good_extension(file, accepted_extensions)

        file_name = str(int(time())) + "_" + str(file.name.split("/")[-1])
        storage_clients(self.api_settings_amazon)["speech"].meta.client.upload_fileobj(
            Fileobj = new_file, 
            Bucket = self.bucket_name, 
            Key= file_name 
        )

        headers = {
            "authorization" : f"Token {self.api_key}",
            "content-type" : f"application/json"
        }

        data = {
            "url": f"{self.storage_url}{file_name}"
        }

        data_config = {
            "language" : language,
            "callback" : self.webhook_url,
            "punctuate" : "true",
            "diarize": "true",
            "profanity_filter": "false"
        }
        if profanity_filter:
            data_config.update({
                "profanity_filter" : "true"
            })

        if not language:
            del data_config["language"]
            data_config.update({
                "detect_language" : "true"
            })
        for key,value in data_config.items():
            self.url = (
                f"{self.url}&{key}={value}"
                if "?" in self.url
                else f"{self.url}?{key}={value}"
            )

        response = requests.post(self.url, headers=headers, json=data)
        result = response.json()
        if response.status_code != 200:
            # delete audio file
            storage_clients(self.api_settings_amazon)["speech"].meta.client.delete_object(
                    Bucket= self.bucket_name, 
                    Key= file_name
                )
            raise ProviderException(f"{result.get('err_code')}: {result.get('err_msg')}")

        transcribe_id = response.json()["request_id"]
        transcribe_id = (
            f"{transcribe_id}1" if data_config["profanity_filter"] == "true"
            else f"{transcribe_id}0"
        )
        return AsyncLaunchJobResponseType(
            provider_job_id= f"{transcribe_id}EdenAI{file_name}"
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        
        provider_job_id, file_name = provider_job_id.split("EdenAI")
        profanity = provider_job_id[-1]
        provider_job_id = provider_job_id[:-1]
        # Getting results from webhook.site
        original_response, response_status = check_webhook_result(provider_job_id, self.api_settings)
        if original_response is None :
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        # delete audio file
        storage_clients(self.api_settings_amazon)["speech"].meta.client.delete_object(
                Bucket= self.bucket_name, 
                Key= file_name
            )
        if response_status != 200:
            raise ProviderException(original_response)
        
        text = ""
        diarization_entries = []
        speakers = set()
        
        if original_response.get("err_code"):
            print(original_response)
            raise ProviderException(f"{original_response.get('err_code')}: {original_response.get('err_msg')}")

        print(json.dumps(original_response, indent=2))
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
        if int(profanity) == 1:
            diarization.error_message = ("Profanity Filter converts profanity to the nearest "
            "recognized non-profane word or removes it from the transcript completely")
        standardized_response = SpeechToTextAsyncDataClass(text=text.strip(), diarization=diarization)
        return AsyncResponseType(
            original_response=original_response,
            standardized_response= standardized_response,
            provider_job_id= provider_job_id
        )
