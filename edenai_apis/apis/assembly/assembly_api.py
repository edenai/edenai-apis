from io import BufferedReader
from pathlib import Path
from typing import List, Optional
import requests
from time import time
from edenai_apis.features import ProviderInterface, AudioInterface
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

from apis.amazon.config import storage_clients
from edenai_apis.utils.upload_s3 import upload_file_to_s3
from .helper import language_matches

class AssemblyApi(ProviderInterface, AudioInterface):
    provider_name = "assembly"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["assembly_key"]
        self.url = self.api_settings["url"]
        self.url_upload_file = f"{self.url}/upload"
        self.url_transcription = f"{self.url}/transcript"

        self.bucket_name = self.api_settings["bucket"]
        self.bucket_region = self.api_settings["region_name"]
        self.storage_url = self.api_settings["storage_url"]
        self.api_settings_amazon = load_provider(ProviderDataEnum.KEY, "amazon")


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

        if language and "-" in language:
            language = language_matches[language]
            
        #upload file to server
        header = {"authorization": self.api_key}
        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )
        data = {
            "audio_url" : f"{content_url}",
            "language_code" : language,
            "speaker_labels" : True,
            "filter_profanity" : profanity_filter
            }
        if vocabulary:
            data.update({"word_boost": vocabulary})
        if not language:
            del data["language_code"]
            data.update({
                "language_detection" : True
            })

        # if an option is not available for a language like 'speaker_labels' with french, we remove it
        launch_transcription = False
        trials = 10
        while not launch_transcription:
            trials -=1
            # launch transcription
            response = requests.post(self.url_transcription, json=data, headers=header)
            if response.status_code != 200:
                error = response.json().get("error")
                if "not available in this language" in error:
                    parameter = error.split(":")[1].strip()
                    del data[parameter]
                else:
                    raise ProviderException(response.json().get("error"))
            else:
                launch_transcription = True

        transcribe_id = f"{response.json()['id']}EdenAI{file_name}"
        return AsyncLaunchJobResponseType(
            provider_job_id= transcribe_id
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:

        provider_job_id, file_name = provider_job_id.split("EdenAI")
        headers = {
            "authorization" : self.api_key
        }

        response= requests.get(url= f"{self.url_transcription}/{provider_job_id}",
            headers=headers
        )

        if response.status_code != 200:
            raise ProviderException(response.json().get("error"))

        diarization_entries = []
        speakers = {}
        index_speaker  = 0

        original_response = response.json()
        status = original_response["status"]
        if status == "error":
            raise ProviderException(original_response)
        if status != "completed":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](provider_job_id=provider_job_id)

        #diarization
        if original_response.get("utterances") and len(original_response["utterances"]) > 0:
            for line in original_response["utterances"]:
                words = line.get("words", [])
                if line["speaker"] not in speakers:
                    index_speaker+=1
                    speaker_tag = index_speaker
                    speakers[line["speaker"]] = index_speaker
                elif line["speaker"] in speakers:
                    speaker_tag = speakers[line["speaker"]]
                for word in words:
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            speaker= speaker_tag,
                            segment= word["text"],
                            start_time= str(word["start"]/1000),
                            end_time= str(word["end"]/1000),
                            confidence= word["confidence"]
                        )
                    )

        diarization = SpeechDiarization(total_speakers= len(speakers), entries= diarization_entries)
        if len(speakers) == 0:
            diarization.error_message = "Speaker diarization not available for the data specified"
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response = original_response,
            standardized_response = SpeechToTextAsyncDataClass(
                text = original_response['text'],
                diarization= diarization
            ),
            provider_job_id = provider_job_id
        )
