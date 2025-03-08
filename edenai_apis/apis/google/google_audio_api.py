import base64
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Optional

import googleapiclient.discovery
from google.cloud import storage, texttospeech
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from edenai_apis.apis.google.google_helpers import (
    generate_tts_params,
    get_right_audio_support_and_sampling_rate,
    handle_google_call,
)
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.ssml import is_ssml
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3


class GoogleAudioApi(AudioInterface):
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
        client = SpeechClient()

        try:
            features = cloud_speech.RecognitionFeatures(**provider_params)
        except ValueError as err:
            """Wrong config are set by users"""
            raise ProviderException(str(err), code=400)

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[language],
            model=model or "long",
            features=features,
        )
        file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

        request = cloud_speech.BatchRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
            config=config,
            files=[file_metadata],
            recognition_output_config=cloud_speech.RecognitionOutputConfig(
                inline_response_config=cloud_speech.InlineOutputConfig(),
            ),
        )

        operation = handle_google_call(client.batch_recognize, request=request)

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
        if original_response.get("done"):
            result = list(original_response["response"]["results"].values())[0]
            for entry in (result.get("transcript", {}) or {}).get("results", []) or []:
                alternatives = entry.get("alternatives")
                if not alternatives or not isinstance(alternatives, list):
                    continue
                alternative = alternatives[0].get("transcript")
                if not alternative:
                    continue
                text += (", " if text else "") + alternative.strip()

            standardized_response = SpeechToTextAsyncDataClass(
                text=text, diarization=SpeechDiarization(total_speakers=0)
            )

            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )
        return AsyncPendingResponseType(provider_job_id=provider_job_id)
