import base64
import uuid
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Optional

import googleapiclient.discovery
from edenai_apis.apis.google.google_helpers import (
    generate_tts_params,
    get_encoding_and_sample_rate,
    get_right_audio_support_and_sampling_rate,
    handle_google_call,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
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

from google.cloud import speech, storage, texttospeech

from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class GoogleAudioApi(AudioInterface):
    def audio__text_to_speech(
        self,
        language: str
        ,
        text: str,
        option: str,
        voice_id: str,
        audio_format: str,
        speaking_rate: int,
        speaking_pitch: int,
        speaking_volume: int,
        sampling_rate: int,
    ) -> ResponseType[TextToSpeechDataClass]:
        voice_type = 1

        client = texttospeech.TextToSpeechClient()

        if is_ssml(text):
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(language_code=language, name=voice_id)

        ext, audio_format = get_right_audio_support_and_sampling_rate(
            audio_format, texttospeech.AudioEncoding._member_names_
        )

        audio_config_params = generate_tts_params(
            speaking_rate, speaking_pitch, speaking_volume
        )

        audio_config_params.update(
            {"audio_encoding": getattr(texttospeech.AudioEncoding, audio_format)}
        )

        if sampling_rate:
            audio_config_params.update({"sample_rate_hertz": sampling_rate})

        audio_config = texttospeech.AudioConfig(**audio_config_params)

        # Getting response of API
        payload = {
            "request": {
                "input": input_text,
                "voice": voice,
                "audio_config": audio_config,
            }
        }
        response = handle_google_call(client.synthesize_speech, **payload)

        audio_content = BytesIO(response.audio_content)

        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url=resource_url
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

        payload = {
            "parent": parent,
            "phrase_set_id": phrase_set_id,
            "phrase_set": {"boost": 10, "phrases": phrases},
        }
        phrase_set_response = handle_google_call(adaptation_client.create_phrase_set, **payload)
        return phrase_set_response.name

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
        provider_params = dict()
    ) -> AsyncLaunchJobResponseType:
        export_format, channels, _ = audio_attributes

        # check language
        if not language:
            raise LanguageException("Language not provided")

        audio_name = str(int(time())) + Path(file).stem + "." + export_format
        # Upload file to google cloud
        storage_client: storage.Client = self.clients["storage"]
        bucket_name = "audios-speech2text"
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(audio_name)
        blob.upload_from_filename(file)

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
            "audio_channel_count": int(channels),
            "enable_separate_recognition_per_channel": True,
            "diarization_config": diarization,
            "profanity_filter": profanity_filter,
            "enable_word_confidence": True,
            "enable_automatic_punctuation": True,
            "enable_spoken_punctuation": True,
            **provider_params
        }

        encoding, sampling = get_encoding_and_sample_rate(
            export_format.replace("audio/", "")
        )

        if encoding:
            params.update(
                {"encoding": getattr(speech.RecognitionConfig.AudioEncoding, encoding)}
            )
        if sampling:
            params.update({"sample_rate_hertz": sampling})

        # create custum vocabulary phrase_set
        if vocabulary:
            name = self._create_vocabulary(vocabulary)
            speech_adaptation = speech.SpeechAdaptation(phrase_set_references=[name])
            params.update({"adaptation": speech_adaptation})

        config = speech.RecognitionConfig(**params)
        payload = {
            "config": config,
            "audio": audio
        }
        operation = handle_google_call(client.long_running_recognize, **payload)
        
        operation_name = operation.operation.name
        return AsyncLaunchJobResponseType(provider_job_id=operation_name)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        service = googleapiclient.discovery.build("speech", "v1")
        service_request_ = service.operations().get(name=provider_job_id)

        original_response = handle_google_call(service_request_.execute)
        
        if (error_message := original_response.get("error")) is not None:
            raise ProviderException(error_message)

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

                try:
                    words_info = result["alternatives"][0]["words"]
                except KeyError:
                    words_info = []

                if words_info == []:
                    error_message = (
                        "Provider has returned an empty response, try to convert your audio file to a 'wav' format, "
                        "or try to put the 'convert_to_wav' parameter to true"
                    )
                    raise ProviderException(error_message)
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
