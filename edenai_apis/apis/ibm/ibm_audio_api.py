import base64
from io import BufferedReader
from typing import List, Optional

from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.utils.audio import audio_features_and_support, file_with_good_extension
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)

from .config import audio_voices_ids


class IbmAudioApi(AudioInterface):
    def audio__text_to_speech(
        self, language: str, text: str, option: str
    ) -> ResponseType[TextToSpeechDataClass]:
        """
        :param language:    String that contains language name 'fr-FR', 'en-US', 'es-EN'
        :param text:        String that contains text to transform
        :param option:      String that contains option of voice(MALE, FEMALE)
        :return:
        """

        option = option.upper()
        # Formatting (option, language) to voice id supported by IBM API
        voiceid = audio_voices_ids[language][option]
        # if one model is not supported for a language
        if not voiceid:
            option_supported = "MALE" if option == "FEMALE" else "FEMALE"
            raise ProviderException(
                f"Only {option_supported} voice is available for the {language} language code"
            )
        try:
            response = (
                self.clients["texttospeech"]
                .synthesize(text=text, accept="audio/mp3", voice=voiceid)
                .get_result()
            )
        except Exception as excp:
            raise ProviderException(excp)
        audio = base64.b64encode(response.content).decode("utf-8")
        voice_type = 1

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type
        )

        return ResponseType[TextToSpeechDataClass](
            original_response={}, standardized_response=standardized_response
        )


    @audio_features_and_support #add audio_attributes to file
    def audio__speech_to_text_async__launch_job(
        self,
        file: BufferedReader,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple
    ) -> AsyncLaunchJobResponseType:
       
        export_format, channels, frame_rate = audio_attributes

        language_audio = language
        audio_config = {
            "audio": file,
            "content_type": "audio/" + export_format,
            "speaker_labels": True,
            "profanity_filter": profanity_filter,
        }
        audio_config.update({"rate": frame_rate})
        if language_audio:
            audio_config.update({"model": f"{language_audio}_NarrowbandModel"})
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
                        speaker=word_info["speaker"] + 1,
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
