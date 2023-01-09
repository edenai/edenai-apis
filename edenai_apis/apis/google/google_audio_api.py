import base64
import uuid
from io import BufferedReader
from pathlib import Path
from time import time
from typing import List, Optional

import googleapiclient.discovery
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.utils.audio import file_with_good_extension
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)

from google.cloud import speech, storage, texttospeech


class GoogleAudioApi(AudioInterface):
    def audio__text_to_speech(
        self, language: str, text: str, option: str
    ) -> ResponseType[TextToSpeechDataClass]:
        voice_type = 1
        ssml_gender = None

        if language in ["da-DK", "pt-BR", "es-ES"] and option == "MALE":
            option = "FEMALE"
            voice_type = 0

        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)

        if option == "FEMALE":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        else:
            ssml_gender = texttospeech.SsmlVoiceGender.MALE

        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            ssml_gender=ssml_gender,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Getting response of API
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        audio = base64.b64encode(response.audio_content).decode("utf-8")

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={},
            standardized_response=standardized_response,
        )

    def _create_vocabulary(self, list_vocabs: list):
        adaptation_client = speech.AdaptationClient()
        parent = f"projects/{self.project_id}/locations/global"
        phrases = [{"value": value} for value in list_vocabs]
        phrase_set_id = str(uuid.uuid4())
        try:
            phrase_set_response = adaptation_client.create_phrase_set(
                {
                    "parent": parent,
                    "phrase_set_id": phrase_set_id,
                    "phrase_set": {"boost": 10, "phrases": phrases},
                }
            )
        except Exception as exc:
            raise ProviderException(str(exc)) from exc
        return phrase_set_response.name

    def audio__speech_to_text_async__launch_job(
        self,
        file: BufferedReader,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
    ) -> AsyncLaunchJobResponseType:

        # check language
        if not language:
            raise LanguageException("Language not provided")

        # check file extension, and convert if not supported
        accepted_extensions = [
            "flac",
            "mp3",
            "wav",
            "ulaw",
            "amr",
            "amr-wb",
            "opus",
            "webm",
            "spx",
        ]
        file, export_format, channels, frame_rate = file_with_good_extension(
            file, accepted_extensions
        )

        audio_name = str(int(time())) + Path(file.name).stem + "." + export_format
        print(audio_name)
        # Upload file to google cloud
        storage_client: storage.Client = self.clients["storage"]
        bucket_name = "audios-speech2text"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(audio_name)
        blob.upload_from_file(file)

        gcs_uri = f"gs://{bucket_name}/{audio_name}"
        # Launch file transcription
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        diarization = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=speakers,
        )

        params = {
            "language_code": language,
            "audio_channel_count": channels,
            "enable_separate_recognition_per_channel": True,
            "diarization_config": diarization,
            "profanity_filter": profanity_filter,
            "enable_word_confidence": True,
            "enable_automatic_punctuation": True,
            "enable_spoken_punctuation": True,
        }
        if export_format == "mp3":
            params.update({"sample_rate_hertz": 16000})

        # create custum vocabulary phrase_set
        if vocabulary:
            name = self._create_vocabulary(vocabulary)
            speech_adaptation = speech.SpeechAdaptation(phrase_set_references=[name])
            params.update({"adaptation": speech_adaptation})

        config = speech.RecognitionConfig(**params)
        try:
            operation = client.long_running_recognize(config=config, audio=audio)
        except Exception as exc:
            error = str(exc)
            if "bad encoding" in error:
                raise ProviderException("Could not decode audio file, bad file encoding")
            raise ProviderException(error)
        operation_name = operation.operation.name
        return AsyncLaunchJobResponseType(provider_job_id=operation_name)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        service = googleapiclient.discovery.build("speech", "v1")
        service_request_ = service.operations().get(name=provider_job_id)
        original_response = service_request_.execute()

        if original_response.get("error") is not None:
            raise ProviderException(original_response["error"])

        text = ""
        diarization = SpeechDiarization(total_speakers=0, entries=[])
        if original_response.get("done"):
            if original_response["response"].get("results"):
                text = ", ".join(
                    [
                        entry["alternatives"][0]["transcript"].strip()
                        if entry["alternatives"][0].get("transcript")
                        else ""
                        for entry in original_response["response"]["results"]
                    ]
                )

                diarization_entries = []
                result = original_response["response"]["results"][-1]
                words_info = result["alternatives"][0]["words"]
                speakers = set()

                for word_info in words_info:
                    speakers.add(word_info["speakerTag"])
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            segment=word_info["word"],
                            speaker=word_info["speakerTag"],
                            start_time=word_info["startTime"][:-1],
                            end_time=word_info["endTime"][:-1],
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
        return AsyncPendingResponseType(provider_job_id=provider_job_id)
