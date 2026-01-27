import base64
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Optional

import aiofiles
import googleapiclient.discovery
from gcloud.aio.storage import Storage as AsyncStorage
from google.cloud import storage, texttospeech
from google.cloud.speech_v2 import SpeechAsyncClient, SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from edenai_apis.apis.google.google_helpers import (
    generate_tts_params,
    get_right_audio_support_and_sampling_rate,
    handle_google_call,
    ahandle_google_call,
)
from edenai_apis.utils.conversion import convert_pitch_from_percentage_to_semitones
from edenai_apis.utils.file_handling import FileHandler
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.features.audio.tts import TtsDataClass
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.ssml import is_ssml
from edenai_apis.utils.tts import get_tts_config
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
    aupload_file_bytes_to_s3,
)


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

    async def audio__atext_to_speech(
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

        payload = {
            "request": {
                "input": input_text,
                "voice": voice,
                "audio_config": audio_config,
            }
        }
        async with texttospeech.TextToSpeechAsyncClient() as client:
            response = await ahandle_google_call(client.synthesize_speech, **payload)

        audio_content = BytesIO(response.audio_content)

        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = await aupload_file_bytes_to_s3(
            audio_content, f".{ext}", USER_PROCESS
        )

        standardized_response = TextToSpeechDataClass(
            audio=audio, voice_type=voice_type, audio_resource_url=resource_url
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={},
            standardized_response=standardized_response,
        )

    async def audio__atts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using Google Cloud TTS API (async version).

        Args:
            text: The text to convert to speech
            model: The TTS model type (e.g., "Standard", "Wavenet", "Neural2").
                   Defaults to "Standard"
            voice: The voice ID (e.g., "en-US-Standard-A", "en-US-Wavenet-D").
                   Defaults to "en-US-Standard-A"
            audio_format: Audio format (mp3, wav, ogg). Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0). Defaults to 1.0
            speaking_pitch: Pitch adjustment (-100 to 100, 0 = normal)
            speaking_volume: Volume adjustment (-100 to 100, 0 = normal)
            provider_params: Provider-specific settings:
                - sampling_rate: Audio sampling rate in Hz
                - language_code: Language code for Gemini TTS (default: "en-US")
        """
        provider_params = provider_params or {}
        config = get_tts_config("google")

        # Normalize model and voice to lowercase
        resolved_model = model.lower() if model else None
        resolved_voice = voice.lower() if voice else None

        # Check if using Gemini TTS model
        is_gemini_model = resolved_model and resolved_model.startswith("gemini-")

        if is_gemini_model:
            # Gemini TTS uses simple voice names like "kore", "puck", etc.
            gemini_voices = [v for v in config["voices"] if "-" not in v]
            if resolved_voice:
                # Validate voice is a Gemini voice
                if resolved_voice not in gemini_voices:
                    raise ProviderException(
                        f"Voice '{voice}' is not supported for Gemini model '{model}'. "
                        f"Supported voices: {', '.join(gemini_voices)}"
                    )
            else:
                resolved_voice = gemini_voices[0] if gemini_voices else "kore"
            language_code = provider_params.get("language_code", "en-US")

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=resolved_voice,
                model_name=resolved_model,
            )
        else:
            # Standard/WaveNet/Neural2/Chirp voices use format like "en-us-standard-a"
            # Default to the default model if none specified
            if not resolved_model:
                resolved_model = config["default_model"]

            model_pattern = f"-{resolved_model}-"

            if resolved_voice:
                # Validate voice matches the model (voice format: "en-us-standard-a")
                voice_parts = resolved_voice.split("-")
                if len(voice_parts) >= 3:
                    voice_model = voice_parts[2]  # e.g., "standard", "wavenet", "neural2"
                    if voice_model != resolved_model:
                        raise ProviderException(
                            f"Voice '{voice}' is not compatible with model '{model or config['default_model']}'. "
                            f"The voice uses '{voice_model}' model. "
                            f"Use a '{resolved_model}' voice or change the model to '{voice_model}'.",
                            code=400
                        )
            else:
                # No voice specified - find first matching English voice for the model
                resolved_voice = next(
                    (
                        v
                        for v in config["voices"]
                        if model_pattern in v and v.startswith("en-")
                    ),
                    next(
                        (v for v in config["voices"] if model_pattern in v),
                        config["default_voice"],
                    ),
                )

            # Extract language code from voice ID (e.g., "en-us-wavenet-d" -> "en-us")
            parts = resolved_voice.split("-")
            if len(parts) >= 2:
                language_code = f"{parts[0]}-{parts[1]}"
            else:
                language_code = "en-US"

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code, name=resolved_voice
            )

        # Build input (support SSML if text starts with <speak>)
        if is_ssml(text):
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)

        # Get audio format mapping
        ext, resolved_audio_format = get_right_audio_support_and_sampling_rate(
            audio_format, texttospeech.AudioEncoding._member_names_
        )

        # Build audio config
        audio_config_params = {
            "audio_encoding": getattr(texttospeech.AudioEncoding, resolved_audio_format)
        }

        # Apply speed (speaking_rate: 0.25 to 4.0, where 1.0 is normal) - not supported for Gemini
        if speed is not None and not is_gemini_model:
            audio_config_params["speaking_rate"] = max(0.25, min(4.0, speed))

        # Apply pitch and volume (not supported for Gemini)
        if not is_gemini_model:
            if speaking_pitch is not None:
                audio_config_params["pitch"] = (
                    convert_pitch_from_percentage_to_semitones(speaking_pitch)
                )
            if speaking_volume is not None:
                # Volume gain in dB (-96 to 16)
                audio_config_params["volume_gain_db"] = max(
                    -96, min(16, speaking_volume * 6 / 100)
                )
        if provider_params.get("sampling_rate") is not None:
            audio_config_params["sample_rate_hertz"] = provider_params["sampling_rate"]

        audio_config = texttospeech.AudioConfig(**audio_config_params)

        payload = {
            "request": {
                "input": input_text,
                "voice": voice_params,
                "audio_config": audio_config,
            }
        }

        async with texttospeech.TextToSpeechAsyncClient() as client:
            response = await ahandle_google_call(client.synthesize_speech, **payload)

        audio_content = BytesIO(response.audio_content)
        resource_url = await aupload_file_bytes_to_s3(
            audio_content, f".{ext}", USER_PROCESS
        )

        standardized_response = TtsDataClass(audio_resource_url=resource_url)
        return ResponseType[TtsDataClass](
            original_response={},
            standardized_response=standardized_response,
        )

    def audio__tts(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: Optional[float] = None,
        speaking_pitch: Optional[int] = None,
        speaking_volume: Optional[int] = None,
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> ResponseType[TtsDataClass]:
        """Convert text to speech using Google Cloud TTS API (sync version).

        Args:
            text: The text to convert to speech
            model: The TTS model type (e.g., "Standard", "Wavenet", "Neural2").
                   Defaults to "Standard"
            voice: The voice ID (e.g., "en-US-Standard-A", "en-US-Wavenet-D").
                   Defaults to "en-US-Standard-A"
            audio_format: Audio format (mp3, wav, ogg). Defaults to "mp3"
            speed: Speech speed (0.25 to 4.0). Defaults to 1.0
            speaking_pitch: Pitch adjustment (-100 to 100, 0 = normal)
            speaking_volume: Volume adjustment (-100 to 100, 0 = normal)
            provider_params: Provider-specific settings:
                - sampling_rate: Audio sampling rate in Hz
                - language_code: Language code for Gemini TTS (default: "en-US")
        """
        provider_params = provider_params or {}
        config = get_tts_config("google")

        # Normalize model and voice to lowercase
        resolved_model = model.lower() if model else None
        resolved_voice = voice.lower() if voice else None

        # Check if using Gemini TTS model
        is_gemini_model = resolved_model and resolved_model.startswith("gemini-")

        if is_gemini_model:
            # Gemini TTS uses simple voice names like "kore", "puck", etc.
            gemini_voices = [v for v in config["voices"] if "-" not in v]
            if resolved_voice:
                # Validate voice is a Gemini voice
                if resolved_voice not in gemini_voices:
                    raise ProviderException(
                        f"Voice '{voice}' is not supported for Gemini model '{model}'. "
                        f"Supported voices: {', '.join(gemini_voices)}"
                    )
            else:
                resolved_voice = gemini_voices[0] if gemini_voices else "kore"
            language_code = provider_params.get("language_code", "en-US")

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=resolved_voice,
                model_name=resolved_model,
            )
        else:
            # Standard/WaveNet/Neural2/Chirp voices use format like "en-us-standard-a"
            # Default to the default model if none specified
            if not resolved_model:
                resolved_model = config["default_model"]

            model_pattern = f"-{resolved_model}-"

            if resolved_voice:
                # Validate voice matches the model (voice format: "en-us-standard-a")
                voice_parts = resolved_voice.split("-")
                if len(voice_parts) >= 3:
                    voice_model = voice_parts[2]  # e.g., "standard", "wavenet", "neural2"
                    if voice_model != resolved_model:
                        raise ProviderException(
                            f"Voice '{voice}' is not compatible with model '{model or config['default_model']}'. "
                            f"The voice uses '{voice_model}' model. "
                            f"Use a '{resolved_model}' voice or change the model to '{voice_model}'.",
                            code=400
                        )
            else:
                # No voice specified - find first matching English voice for the model
                resolved_voice = next(
                    (
                        v
                        for v in config["voices"]
                        if model_pattern in v and v.startswith("en-")
                    ),
                    next(
                        (v for v in config["voices"] if model_pattern in v),
                        config["default_voice"],
                    ),
                )

            # Extract language code from voice ID (e.g., "en-us-wavenet-d" -> "en-us")
            parts = resolved_voice.split("-")
            if len(parts) >= 2:
                language_code = f"{parts[0]}-{parts[1]}"
            else:
                language_code = "en-US"

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code, name=resolved_voice
            )

        # Build input (support SSML if text starts with <speak>)
        if is_ssml(text):
            input_text = texttospeech.SynthesisInput(ssml=text)
        else:
            input_text = texttospeech.SynthesisInput(text=text)

        # Get audio format mapping
        ext, resolved_audio_format = get_right_audio_support_and_sampling_rate(
            audio_format, texttospeech.AudioEncoding._member_names_
        )

        # Build audio config
        audio_config_params = {
            "audio_encoding": getattr(texttospeech.AudioEncoding, resolved_audio_format)
        }

        # Apply speed (speaking_rate: 0.25 to 4.0, where 1.0 is normal) - not supported for Gemini
        if speed is not None and not is_gemini_model:
            audio_config_params["speaking_rate"] = max(0.25, min(4.0, speed))

        # Apply pitch and volume (not supported for Gemini)
        if not is_gemini_model:
            if speaking_pitch is not None:
                audio_config_params["pitch"] = (
                    convert_pitch_from_percentage_to_semitones(speaking_pitch)
                )
            if speaking_volume is not None:
                # Volume gain in dB (-96 to 16)
                audio_config_params["volume_gain_db"] = max(
                    -96, min(16, speaking_volume * 6 / 100)
                )
        if provider_params.get("sampling_rate") is not None:
            audio_config_params["sample_rate_hertz"] = provider_params["sampling_rate"]

        audio_config = texttospeech.AudioConfig(**audio_config_params)

        payload = {
            "request": {
                "input": input_text,
                "voice": voice_params,
                "audio_config": audio_config,
            }
        }

        client = texttospeech.TextToSpeechClient()
        response = handle_google_call(client.synthesize_speech, **payload)

        audio_content = BytesIO(response.audio_content)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)

        standardized_response = TtsDataClass(audio_resource_url=resource_url)
        return ResponseType[TtsDataClass](
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

    async def audio__aspeech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: Optional[tuple] = None,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        provider_params = provider_params or {}
        export_format, channels, _ = audio_attributes
        file_handler = FileHandler()
        file_wrapper = None

        # check language
        if not language:
            raise LanguageException("Language not provided")

        try:
            bucket_name = "audios-speech2text"

            if file_url and not file:
                # Download file from URL and get bytes directly
                file_wrapper = await file_handler.download_file(file_url)
                file_content = await file_wrapper.get_bytes()
                file_ext = file_wrapper.file_info.file_extension or "wav"
                audio_name = str(int(time())) + "_audio." + file_ext
            elif file:
                async with aiofiles.open(file, "rb") as f:
                    file_content = await f.read()
                audio_name = str(int(time())) + Path(file).stem + "." + export_format
            else:
                raise ProviderException(
                    "Either file or file_url must be provided", code=400
                )

            # Upload to GCS using async storage client
            async with AsyncStorage() as async_storage:
                await async_storage.upload(
                    bucket_name, audio_name, file_content
                )

            gcs_uri = f"gs://{bucket_name}/{audio_name}"

            # Launch file transcription using async client
            try:
                features = cloud_speech.RecognitionFeatures(**provider_params)
            except ValueError as err:
                raise ProviderException(str(err), code=400) from err

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

            async with SpeechAsyncClient() as speech_async_client:
                operation = await ahandle_google_call(
                    speech_async_client.batch_recognize, request=request
                )

            operation_name = operation.operation.name
            return AsyncLaunchJobResponseType(provider_job_id=operation_name)
        finally:
            if file_wrapper:
                file_wrapper.close_file()
