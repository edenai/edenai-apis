import base64
import json
import urllib
import uuid
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Optional

import requests
from botocore.exceptions import BotoCoreError, ClientError

from edenai_apis.apis.amazon.helpers import (
    generate_right_ssml_text,
    get_right_audio_support_and_sampling_rate,
    handle_amazon_call,
)
from edenai_apis.features.audio import TextToSpeechAsyncDataClass
from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.ssml import is_ssml
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import (
    URL_LONG_PERIOD,
    USER_PROCESS,
    get_cloud_front_file_url,
    get_s3_file_url,
    s3_client_load,
    upload_file_bytes_to_s3,
)

from .config import audio_voices_ids, storage_clients


class AmazonAudioApi(AudioInterface):
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
        _, voice_id_name, engine = voice_id.split("_")
        engine = engine.lower()

        params = {"Engine": engine, "VoiceId": voice_id_name, "OutputFormat": "mp3"}

        text = generate_right_ssml_text(
            text, speaking_rate, speaking_pitch, speaking_volume
        )

        ext, audio_format, sampling = get_right_audio_support_and_sampling_rate(
            audio_format, sampling_rate
        )

        params_update = {"OutputFormat": audio_format, "Text": text}
        if sampling:
            params_update["SampleRate"] = str(sampling)

        params.update({**params_update})

        if is_ssml(text):
            params["TextType"] = "ssml"

        response = handle_amazon_call(
            self.clients["texttospeech"].synthesize_speech, **params
        )

        audio_content = BytesIO(response["AudioStream"].read())

        # convert 'StreamBody' to b64
        audio_file = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)

        standardized_response = TextToSpeechDataClass(
            audio=audio_file, voice_type=voice_type, audio_resource_url=resource_url
        )
        return ResponseType[TextToSpeechDataClass](
            original_response={}, standardized_response=standardized_response
        )

    # Speech to text async
    def _upload_audio_file_to_amazon_server(
        self, file_path: str, file_name: str
    ) -> str:
        """
        :param audio_path:  String that contains the audio file path
        :return:            String that contains the filename on the server
        """
        # Store file in an Amazon server
        # filename = str(int(time())) + "_" + str(file_name)
        filename = str(uuid.uuid4())
        self.storage_clients["speech"].meta.client.upload_file(
            file_path, self.api_settings["bucket"], filename
        )

        return filename

    def _create_vocabulary(self, language: str, list_vocabs: list):
        list_vocabs = ["-".join(vocab.strip().split()) for vocab in list_vocabs]
        vocab_name = str(uuid.uuid4())
        payload = {
            "LanguageCode": language,
            "VocabularyName": vocab_name,
            "Phrases": list_vocabs,
        }
        response = handle_amazon_call(
            self.clients["speech"].create_vocabulary, **payload
        )

        return vocab_name

    def _launch_transcribe(
        self,
        filename: str,
        frame_rate,
        language: str,
        speakers: int,
        vocab_name: Optional[str] = None,
        initiate_vocab: bool = False,
        format: str = "wav",
        provider_params: Optional[dict] = None,
    ):
        provider_params = provider_params or {}
        if speakers < 2:
            speakers = 2
        params = {
            "TranscriptionJobName": filename,
            "Media": {"MediaFileUri": self.api_settings["storage_url"] + filename},
            "LanguageCode": language,
            "Settings": {
                "ShowSpeakerLabels": True,
                "ChannelIdentification": False,
                "MaxSpeakerLabels": speakers,
            },
        }
        if not language:
            del params["LanguageCode"]
            params.update({"IdentifyMultipleLanguages": True})
        if vocab_name:
            params["Settings"].update({"VocabularyName": vocab_name})
            if initiate_vocab:
                params["checked"] = False
                # extention_index = filename.rfind(".")
                filename = f"{filename}_settings.txt"
                self.storage_clients["speech"].meta.client.put_object(
                    Bucket=self.api_settings["bucket"],
                    Body=json.dumps(params).encode(),
                    Key=filename,
                )
                return
        params.update(provider_params)
        try:
            self.clients["speech"].start_transcription_job(**params)
        except KeyError as exc:
            raise ProviderException(str(exc)) from exc

    def _delete_vocabularies(self, vocab_name):
        payload = {"VocabularyName": vocab_name}
        handle_amazon_call(self.clients["speech"].delete_vocabulary, **payload)

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: list,
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        provider_params = provider_params or {}
        export_format, channels, frame_rate = audio_attributes

        filename = self._upload_audio_file_to_amazon_server(
            file, Path(file).stem + "." + export_format
        )
        if vocabulary:
            if language is None:
                raise ProviderException(
                    "Cannot launch with vocabulary when language is auto-detect.",
                    code=400,
                )
            vocab_name = self._create_vocabulary(language, vocabulary)
            self._launch_transcribe(
                filename,
                frame_rate,
                language,
                speakers,
                vocab_name,
                True,
                format=export_format,
                provider_params=provider_params,
            )
            return AsyncLaunchJobResponseType(
                provider_job_id=f"{filename}EdenAI{vocab_name}"
            )

        self._launch_transcribe(
            filename,
            frame_rate,
            language,
            speakers,
            format=export_format,
            provider_params=provider_params,
        )
        return AsyncLaunchJobResponseType(provider_job_id=filename)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        # check custom vocabilory job state
        job_id, *vocab = provider_job_id.split("EdenAI")
        if vocab:  # if vocabilory is used and
            can_use_vocab = True  # if failed, we don't use the vocabulary
            setting_content = self.storage_clients["speech"].meta.client.get_object(
                Bucket=self.api_settings["bucket"], Key=f"{job_id}_settings.txt"
            )
            settings = json.loads(setting_content["Body"].read().decode("utf-8"))
            if not settings[
                "checked"
            ]:  # check if the vocabulary has been created or not
                vocab_name = vocab[0]
                job_vocab_details = self.clients["speech"].get_vocabulary(
                    VocabularyName=vocab_name
                )
                if job_vocab_details["VocabularyState"] == "FAILED":
                    self._delete_vocabularies(vocab_name)
                    can_use_vocab = False
                if job_vocab_details["VocabularyState"] not in ["READY", "FAILED"]:
                    return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                        provider_job_id=provider_job_id
                    )

                self._launch_transcribe(
                    settings["TranscriptionJobName"],
                    "",
                    settings["LanguageCode"],
                    settings["Settings"]["MaxSpeakerLabels"],
                    settings["Settings"]["VocabularyName"] if can_use_vocab else None,
                )
                settings["checked"] = True  # confirm vocabulary creation
                extention_index = job_id.rfind(".")
                index_last = len(job_id) - extention_index
                self.storage_clients["speech"].meta.client.put_object(
                    Bucket=self.api_settings["bucket"],
                    Body=json.dumps(settings).encode(),
                    Key=f"{job_id}_settings.txt",
                )
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                    provider_job_id=provider_job_id
                )

        # check transcribe status
        payload = {"TranscriptionJobName": job_id}
        job_details = handle_amazon_call(
            self.clients["speech"].get_transcription_job, **payload
        )

        job_status = job_details["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status == "COMPLETED":
            # delete vocabulary
            try:
                self._delete_vocabularies(vocab[0])
            except IndexError as ir:  # if no vocabulary was created
                pass

            json_res = job_details["TranscriptionJob"]["Transcript"][
                "TranscriptFileUri"
            ]
            with urllib.request.urlopen(json_res) as url:
                original_response = json.loads(url.read().decode("utf-8"))
                # add metadata to the response (settings, result/subtitle urls etc.)
                original_response.update(job_details)
                # diarization
                diarization_entries = []
                words_info = original_response["results"]["items"]
                speakers = (
                    original_response.get("results", {}).get("speaker_labels", {}) or {}
                ).get("speakers", 0)

                for word_info in words_info:
                    if word_info.get("speaker_label"):
                        if word_info["type"] == "pronunciation":
                            diarization_entries.append(
                                SpeechDiarizationEntry(
                                    segment=word_info["alternatives"][0]["content"],
                                    speaker=int(
                                        word_info["speaker_label"].split("spk_")[1]
                                    )
                                    + 1,
                                    start_time=word_info["start_time"],
                                    end_time=word_info["end_time"],
                                    confidence=word_info["alternatives"][0][
                                        "confidence"
                                    ],
                                )
                            )
                        else:
                            diarization_entries[
                                len(diarization_entries) - 1
                            ].segment = (
                                f"{diarization_entries[len(diarization_entries)-1].segment}"
                                f"{word_info['alternatives'][0]['content']}"
                            )

                standardized_response = SpeechToTextAsyncDataClass(
                    text=original_response["results"]["transcripts"][0]["transcript"],
                    diarization=SpeechDiarization(
                        total_speakers=speakers, entries=diarization_entries
                    ),
                )
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                    provider_job_id=provider_job_id,
                )
        elif job_status == "FAILED":
            # delete vocabulary
            try:
                self._delete_vocabularies(vocab[0])
            except IndexError as ir:  # if no vocabulary was created
                pass

            error = job_details["TranscriptionJob"].get("FailureReason")
            raise ProviderException(error)
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )

    def audio__text_to_speech_async__launch_job(
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
        file_url: str = "",
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        _, voice_id_name, engine = voice_id.split("_")
        engine = engine.lower()

        params = {
            "Engine": engine,
            "VoiceId": voice_id_name,
            "OutputFormat": "mp3",
            "OutputS3BucketName": self.api_settings["users_resource_bucket"],
        }

        text = generate_right_ssml_text(
            text, speaking_rate, speaking_pitch, speaking_volume
        )

        ext, audio_format, sampling = get_right_audio_support_and_sampling_rate(
            audio_format, sampling_rate
        )

        params_update = {"OutputFormat": audio_format, "Text": text}
        if sampling:
            params_update["SampleRate"] = str(sampling)

        params.update({**params_update})

        if is_ssml(text):
            params["TextType"] = "ssml"

        response = handle_amazon_call(
            self.clients["texttospeech"].start_speech_synthesis_task, **params
        )
        synthesis_task = response["SynthesisTask"]
        if synthesis_task["TaskStatus"] == "failed":
            raise ProviderException(
                synthesis_task.get(
                    "TaskStatusReason", "Amazon returned a job status: failed"
                )
            )
        print(synthesis_task["TaskId"])
        return AsyncLaunchJobResponseType(provider_job_id=synthesis_task["TaskId"])

    def audio__text_to_speech_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[TextToSpeechAsyncDataClass]:
        params = {"TaskId": provider_job_id}
        response = handle_amazon_call(
            self.clients["texttospeech"].get_speech_synthesis_task, **params
        )
        synthesis_task = response["SynthesisTask"]
        status = synthesis_task["TaskStatus"]
        if status == "failed":
            raise ProviderException(
                synthesis_task.get(
                    "TaskStatusReason", "Amazon returned a job status: failed"
                )
            )
        elif status == "inProgress" or status == "scheduled":
            return AsyncPendingResponseType[TextToSpeechAsyncDataClass](
                provider_job_id=provider_job_id
            )
        else:
            output_uri = synthesis_task.get("OutputUri", "")
            s3_client_load()
            file_url = get_cloud_front_file_url(
                output_uri.split("/")[-1], URL_LONG_PERIOD
            )
            synthesis_task["OutputUri"] = file_url
            response_file = requests.get(file_url)
            audio_content = BytesIO(response_file.content)
            audio = base64.b64encode(audio_content.read()).decode("utf-8")
            standardized_response = TextToSpeechAsyncDataClass(
                audio_resource_url=file_url, audio=audio, voice_type=1
            )
            return AsyncResponseType[TextToSpeechAsyncDataClass](
                original_response=synthesis_task,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )
