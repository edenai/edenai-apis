import base64
import json
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import List, Optional

import azure.cognitiveservices.speech as speechsdk
import requests
from edenai_apis.features.audio import (
    AudioInterface,
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.utils.audio import retreive_voice_id
from edenai_apis.utils.conversion import convert_pt_date_from_string
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3, upload_file_to_s3

from .config import audio_voice_ids


class MicrosoftAudioApi(AudioInterface):
    def audio__text_to_speech(
        self, language: str, text: str, option: str, settings: dict = {}
    ) -> ResponseType[TextToSpeechDataClass]:
        
        voice_id = retreive_voice_id(self, language, option, settings)

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_settings["speech"]["subscription_key"],
            region=self.api_settings["speech"]["service_region"],
        )

        speech_config.speech_synthesis_voice_name = voice_id
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        # Getting response of API
        # output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        response = speech_synthesizer.speak_text_async(text).get()

        if response.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = response.cancellation_details

            raise ProviderException(
                "error", f"Speech synthesis canceled: {cancellation_details.reason}"
            )

        audio_content = BytesIO(response.audio_data)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, ".mp3", USER_PROCESS)

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url = resource_url
        )

        return ResponseType[TextToSpeechDataClass](
            original_response={}, standardized_response=standardized_response
        )

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes

        # check language
        if not language:
            raise LanguageException("Language not provided")

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(
                file, Path(file).stem + "." + export_format
            )

        headers = self.headers["speech"]
        headers["Content-Type"] = "application/json"

        config = {
            "contentUrls": [content_url],
            "properties": {
                "wordLevelTimestampsEnabled": True,
                "profanityFilterMode": "None",
            },
            "locale": language,
            "displayName": "test batch transcription",
        }
        if int(channels) == 1:
            config["properties"].update(
                {
                    "diarizationEnabled": True,
                }
            )
        if profanity_filter:
            config["properties"].update({"profanityFilterMode": "Masked"})
        # if not profanity_filter:
        #     config["properties"]["profanityFilterMode"] = "Removed"

        response = requests.post(
            url=self.url["speech"], headers=headers, data=json.dumps(config)
        )
        print(response.json())
        if response.status_code == 201:
            result_location = response.headers["Location"]
            provider_id = result_location.split("/")[-1]
            return AsyncLaunchJobResponseType(provider_job_id=provider_id)
        else:
            raise Exception(response.json().get("message"))

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:

        headers = self.headers["speech"]
        response = requests.get(
            url=f'{self.url["speech"]}/{provider_job_id}/files', headers=headers
        )
        original_response = None
        if response.status_code == 200:
            data = response.json()["values"]
            if data:
                files_urls = [
                    entry["links"]["contentUrl"]
                    for entry in data
                    if entry["kind"] == "Transcription"
                ]
                text = ""
                diarization_entries = []
                speakers = set()
                for file_url in files_urls:
                    response = requests.get(file_url, headers=headers)
                    original_response = response.json()
                    if response.status_code != 200:
                        error = original_response.get("message")
                        raise ProviderException(error)
                    if (
                        original_response["combinedRecognizedPhrases"]
                        and len(original_response["combinedRecognizedPhrases"]) > 0
                    ):
                        data = original_response["combinedRecognizedPhrases"][0]
                        text += data["display"]
                        for recognized_status in original_response["recognizedPhrases"]:
                            if recognized_status["recognitionStatus"] == "Success":
                                if "speaker" in recognized_status:
                                    speaker = recognized_status["speaker"]
                                    for word_info in recognized_status["nBest"][0][
                                        "words"
                                    ]:
                                        speakers.add(speaker)
                                        start_time = convert_pt_date_from_string(
                                            word_info["offset"]
                                        )
                                        end_time = (
                                            start_time
                                            + convert_pt_date_from_string(
                                                word_info["duration"]
                                            )
                                        )
                                        diarization_entries.append(
                                            SpeechDiarizationEntry(
                                                segment=word_info["word"],
                                                speaker=speaker,
                                                start_time=str(start_time),
                                                end_time=str(end_time),
                                                confidence=float(
                                                    word_info["confidence"]
                                                ),
                                            )
                                        )

                diarization = SpeechDiarization(
                    total_speakers=len(speakers), entries=diarization_entries
                )
                if len(speakers) == 0:
                    diarization.error_message = "Use mono audio files for diarization"

                standardized_response = SpeechToTextAsyncDataClass(
                    text=text, diarization=diarization
                )
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                    provider_job_id=provider_job_id,
                )
            else:
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                    provider_job_id=provider_job_id
                )
        else:
            error = response.json().get("message")
            raise ProviderException(error)
