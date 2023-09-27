
from pathlib import Path
from typing import Dict
import requests
import json
from time import time
import uuid
from edenai_apis.features import ProviderInterface, AudioInterface
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import SpeechToTextAsyncDataClass
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from apis.amazon.helpers import check_webhook_result

from edenai_apis.utils.upload_s3 import upload_file_to_s3

from .helper import language_matches


class GladiaApi(ProviderInterface, AudioInterface):
    provider_name = "gladia"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name, api_keys = api_keys)
        self.api_key = self.api_settings["gladia_key"]
        self.url = "https://api.gladia.io/audio/text/audio-transcription/"
        self.webhook_settings = load_provider(ProviderDataEnum.KEY, "webhooksite")
        self.webhook_token = self.webhook_settings["webhook_token"]


    def audio__speech_to_text_async__launch_job(
        self, 
        file: str,
        language: str, 
        speakers: int, 
        profanity_filter: bool, 
        vocabulary: list,
        audio_attributes: tuple,
        model : str = None,
        file_url: str = "",
        provider_params: dict = dict(),
        ) -> AsyncLaunchJobResponseType:
        data_job_id = {}
        export_format, channels, frame_rate = audio_attributes

        headers = {
            "x-gladia-key": self.api_key,
        }


        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )

        files = {
            'audio_url': content_url,
        }
        data = {
            'language_behaviour': 'automatic multiple language',
            'toggle_diarization': 'true',
        }

        if language:
            data.update({
                'language': language_matches[language],
                'language_behaviour': 'manual',
            })
        data.update(provider_params)


        response = requests.post('https://api.gladia.io/audio/text/audio-transcription/', headers=headers, files=files, data=data)

        original_response = response.json()
        if response.status_code != 200:
            # error example
            """
            {
            "statusCode": 400,
            "timestamp": "2023-07-05T03:02:05.402Z",
            "path": "/audio/text/audio-transcription/?model=large-v2",
            "message": "You need to specify at least one of these fields: video, video_url",
            "request_id": " "
            }
            """
            raise ProviderException(
                f"{original_response.get('timestamp')} - {original_response.get('statusCode')}: "\
                f"{original_response.get('message')} - request_id: {original_response.get('request_id')}",
                code = response.status_code
            )
        
        job_id = "gladia_stt" + str(uuid.uuid4())
        data_job_id[job_id] = original_response
        requests.post(
            url = f'https://webhook.site/{self.webhook_token}',
            data = json.dumps(data_job_id),
            
            headers = {'content-type':'application/json'})
        return AsyncLaunchJobResponseType(provider_job_id=job_id)
    
    def audio__speech_to_text_async__get_job_result(
        self,
        provider_job_id: str
        ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        
        # Get results from webhooks : 
        # List all webhook results
        # Getting results from webhook.site

        wehbook_result, response_status = check_webhook_result(provider_job_id, self.webhook_settings)

        if response_status != 200:
            raise ProviderException(wehbook_result, code = response_status)
        
        result_object = next(filter(lambda response: provider_job_id in response["content"], wehbook_result), None) \
            if wehbook_result else None

        if not result_object or not result_object.get("content"):
            raise ProviderException("Provider returned an empty response")

        try:
            original_response = json.loads(result_object["content"]).get(provider_job_id, None)
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")

        if original_response is None:
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        diarization_entries = []
        speakers = {}
        index_speaker  = 0
        text = ""
        if "prediction" in original_response:
            for utterance in original_response["prediction"]:
                text_segment = utterance.get("transcription", "")
                time_begin = utterance.get("time_begin", 0)
                time_end = utterance.get("time_end", 0)
                speaker_id = utterance.get("speaker", 0)
                confidence = -1
                text += f"{text_segment} "
                if speaker_id not in speakers:
                    index_speaker += 1
                    speaker_tag = index_speaker
                    speakers[speaker_id] = index_speaker
                elif speaker_id in speakers:
                    speaker_tag = speakers[speaker_id]

                # calculate the average of the confidence of each word
                for word in utterance.get("words", []):
                    confidence += word.get("confidence", 0)

                confidence = confidence / len(utterance.get("words", []))

                diarization_entries.append(
                    SpeechDiarizationEntry(
                        speaker=speaker_tag,
                        segment=text_segment,
                        start_time=str(time_begin),
                        end_time=str(time_end),
                        confidence=confidence
                    )
                )

        diarization = SpeechDiarization(total_speakers=len(speakers), entries=diarization_entries)
        if len(speakers) == 0:
            diarization.error_message = "Speaker diarization not available for the data specified"
            
        standardized_response = SpeechToTextAsyncDataClass(text = text, diarization=diarization)
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )
