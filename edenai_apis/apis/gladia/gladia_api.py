
from io import BufferedReader
from pathlib import Path
from typing import Dict
import requests
import json
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
from apis.amazon.helpers import check_webhook_result

from apis.amazon.config import storage_clients
from edenai_apis.utils.upload_s3 import upload_file_to_s3

from .helper import language_matches


class GladiaApi(ProviderInterface, AudioInterface):
    provider_name = "gladia"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name, api_keys = api_keys)
        self.api_key = self.api_settings["gladia_key"]
        self.url = "https://api.gladia.io/audio/text/audio-transcription/"


    def audio__speech_to_text_async__launch_job(
        self, 
        file: str,
        language: str, 
        speakers: int, 
        profanity_filter: bool, 
        vocabulary: list,
        audio_attributes: tuple,
        model: str,
        file_url: str = "",
        ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes


        headers = {
            "x-gladia-key": f"Token {self.api_key}",
            "content-type" : f"application/json"
        }



        file_name = str(int(time())) + "_" + str(file.split("/")[-1])

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file_name).stem + "." + export_format
            )

        files = {
            'audio_url': (None, content_url),
            'language_behaviour': (None, 'automatic multiple language'),
            'toggle_diarization': (None, 'true'),
        }

        if language:
            files.update({
                'language': (None, language_matches[language]),
                'language_behaviour': (None, 'manual'),
            })



        response = requests.post('https://api.gladia.io/audio/text/audio-transcription/', headers=headers, files=files)

        original_response = response.json()
        if original_response.status_code != 200:
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
            raise ProviderException(f"{original_response.get('timestamp')} - {original_response.get('statusCode')}: {original_response.get('message')} - request_id: {original_response.get('request_id')}")
        else:
            diarization_entries = []
            speakers = {}
            index_speaker  = 0

            if "prediction" in original_response:
                for utterance in original_response["prediction"]:
                    text_segment = utterance.get("transcription", "")
                    time_begin = utterance.get("time_begin", 0)
                    time_end = utterance.get("time_end", 0)
                    speaker_id = utterance.get("speaker", 0)
                    confidence = -1

                    if speaker_id not in speakers:
                        utterance += 1
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


