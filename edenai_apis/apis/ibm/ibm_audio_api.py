import base64
from io import BufferedReader, BytesIO
from typing import List, Optional
from edenai_apis.apis.ibm.ibm_helpers import generate_right_ssml_text, get_right_audio_support_and_sampling_rate

from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.utils.audio import retreive_voice_id
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3

from watson_developer_cloud.watson_service import WatsonApiException

class IbmAudioApi(AudioInterface):
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
        sampling_rate: int
    ) -> ResponseType[TextToSpeechDataClass]:
        """
        :param language:    String that contains language name 'fr-FR', 'en-US', 'es-EN'
        :param text:        String that contains text to transform
        :param option:      String that contains option of voice(MALE, FEMALE)
        :return:
        """
        text = generate_right_ssml_text(text, speaking_rate, speaking_pitch)

        ext, audio_format = get_right_audio_support_and_sampling_rate(audio_format, sampling_rate)

        params = {
            "text": text,
            "accept": audio_format,
            "voice": voice_id
        }

        try:
            response = (
                self.clients["texttospeech"]
                .synthesize(**params)
                .get_result()
            )
        except WatsonApiException as excp:
            raise ProviderException(excp.message)
        
        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)
        
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
        model : str = None,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes
       
        language_audio = language
        
        file_ = open(file, "rb")
        audio_config = {
            "audio": file_,
            "content_type": "audio/" + export_format,
            "speaker_labels": True,
            "profanity_filter": profanity_filter,
        }
        audio_config.update({"rate": int(frame_rate)})
        if language_audio:
            audio_config.update({"model": f"{language_audio}_Telephony"})
            if language_audio == "ja-JP":
                audio_config["model"] = f"{language_audio}_Multimedia"
        response = self.clients["speech"].create_job(**audio_config)
        if response.status_code == 201:
            return AsyncLaunchJobResponseType(provider_job_id=response.result["id"])
        else:
            raise ProviderException("An error occured during ibm api call")

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        response = self.clients["speech"].check_job(provider_job_id)
        status = response.result["status"]
        if status == "completed":
            original_response = response.result["results"]
            data = response.result["results"][0]["results"]

            diarization_entries = []
            speakers = set()

            text = " ".join([entry["alternatives"][0]["transcript"] for entry in data])

            time_stamps = [
                time_stamp
                for entry in data
                for time_stamp in entry["alternatives"][0]["timestamps"]
            ]
            for idx_word, word_info in enumerate(
                original_response[0].get("speaker_labels", [])
            ):
                speakers.add(word_info["speaker"])
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment=time_stamps[idx_word][0],
                        start_time=str(time_stamps[idx_word][1]),
                        end_time=str(time_stamps[idx_word][2]),
                        speaker= int(word_info["speaker"]) + 1,
                        confidence=word_info["confidence"],
                    )
                )
            diarization = SpeechDiarization(
                total_speakers=len(speakers), entries=diarization_entries
            )
            standardized_response = SpeechToTextAsyncDataClass(
                text=text, diarization=diarization
            )
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )

        if status == "failed":
            # Apparently no error message present in response
            # ref: https://cloud.ibm.com/apidocs/speech-to-text?code=python#checkjob
            raise ProviderException

        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
