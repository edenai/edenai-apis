import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import azure.cognitiveservices.speech as speechsdk
import requests

from edenai_apis.apis.microsoft.microsoft_helpers import (
    generate_right_ssml_text,
    get_right_audio_support_and_sampling_rate,
)
from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.utils.conversion import convert_pt_date_from_string
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    LanguageException,
    ProviderException,
)
from edenai_apis.utils.ssml import is_ssml
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import (
    USER_PROCESS,
    upload_file_bytes_to_s3,
    upload_file_to_s3,
)


class MicrosoftAudioApi(AudioInterface):
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
        **kwargs,
    ) -> ResponseType[TextToSpeechDataClass]:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_settings["speech"]["subscription_key"],
            region=self.api_settings["speech"]["service_region"],
        )
        speech_config.speech_synthesis_voice_name = voice_id

        ext, audio_format = get_right_audio_support_and_sampling_rate(
            audio_format, 0, speechsdk.SpeechSynthesisOutputFormat._member_names_
        )
        speech_config.set_speech_synthesis_output_format(
            getattr(speechsdk.SpeechSynthesisOutputFormat, audio_format)
        )

        text = generate_right_ssml_text(
            text, voice_id, speaking_rate, speaking_pitch, speaking_volume
        )

        # Getting response of API
        # output_config = speechsdk.audio.AudioOutputConfig(filename="outputaudio.wav")
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        response = (
            speech_synthesizer.speak_text_async(text).get()
            if not is_ssml(text)
            else speech_synthesizer.speak_ssml_async(text).get()
        )

        if response.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = response.cancellation_details
            raise ProviderException(str(cancellation_details.error_details))

        audio_content = BytesIO(response.audio_data)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url=resource_url
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
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        provider_params = provider_params or {}
        export_format, channels, frame_rate = audio_attributes

        # check language
        if not language:
            raise LanguageException("Language not provided")

        content_url = file_url
        if not content_url:
            content_url = upload_file_to_s3(file, Path(file).stem + "." + export_format)

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

        config.update(provider_params)
        response = requests.post(
            url=self.url["speech"], headers=headers, data=json.dumps(config)
        )
        if response.status_code == 201:
            result_location = response.headers["Location"]
            provider_id = result_location.split("/")[-1]
            return AsyncLaunchJobResponseType(provider_job_id=provider_id)
        else:
            raise ProviderException(
                response.json().get("message"), code=response.status_code
            )

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
            if error:
                if "entity cannot be found" in error:
                    raise AsyncJobException(
                        reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                        code=response.status_code,
                    )
                raise ProviderException(error, code=response.status_code)
            raise ProviderException(response.json(), code=response.status_code)
