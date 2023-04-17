import json
from pathlib import Path
from typing import Optional
import urllib
import uuid
import base64
from io import BufferedReader, BytesIO
from botocore.exceptions import BotoCoreError
from edenai_apis.apis.amazon.helpers import amazon_speaking_rate_converter, generate_right_ssml_text, get_right_audio_support_and_sampling_rate

from edenai_apis.features.audio.audio_interface import AudioInterface
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import (
    TextToSpeechDataClass,
)
from edenai_apis.utils.audio import retreive_voice_id
from edenai_apis.utils.exception import AsyncJobException, AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3, USER_PROCESS

from .config import audio_voices_ids


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
    ) -> ResponseType[TextToSpeechDataClass]:
        
        params = {
            "VoiceId" : voice_id.split("_")[1],
            "OutputFormat": "mp3"
        }

        text, text_type = generate_right_ssml_text(text, speaking_rate, speaking_pitch, speaking_volume)

        ext, audio_format, sampling = get_right_audio_support_and_sampling_rate(audio_format, sampling_rate)

        params_update = {
            "OutputFormat": audio_format,
            "Text": text
        }
        if sampling:
            params_update["SampleRate"] = str(sampling)
        
        params.update({
            **params_update
        })

        if text_type:
            params["TextType"] = "ssml"

        try:
            response = self.clients["texttospeech"].synthesize_speech(**params)
        except Exception as excp:
            raise ProviderException(str(excp))


        audio_content = BytesIO(response["AudioStream"].read())

        # convert 'StreamBody' to b64
        audio_file = base64.b64encode(audio_content.read()).decode("utf-8")
        voice_type = 1

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, f".{ext}", USER_PROCESS)

        standardized_response = TextToSpeechDataClass(
            audio=audio_file, voice_type=voice_type, audio_resource_url = resource_url
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
        try:
            self.clients["speech"].create_vocabulary(
                LanguageCode=language, VocabularyName=vocab_name, Phrases=list_vocabs
            )
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

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
    ):
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
            params.update({"IdentifyLanguage": True})
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
        try:
            self.clients["speech"].start_transcription_job(**params)
        except KeyError as exc:
            raise ProviderException(str(exc)) from exc
        

    def _delete_vocabularies(self, vocab_name):
        try:
            self.clients["speech"].delete_vocabulary(VocabularyName=vocab_name)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc


    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: list,
        audio_attributes: tuple,
        model : str = None,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes


        filename = self._upload_audio_file_to_amazon_server(
            file, Path(file).stem + "." + export_format
        )
        if vocabulary:
            vocab_name = self._create_vocabulary(language, vocabulary)
            self._launch_transcribe(
                filename,
                frame_rate,
                language,
                speakers,
                vocab_name,
                True,
                format=export_format,
            )
            return AsyncLaunchJobResponseType(
                provider_job_id=f"{filename}EdenAI{vocab_name}"
            )

        self._launch_transcribe(
            filename, frame_rate, language, speakers, format=export_format
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
            can_use_vocab = True # if failed, we don't use the vocabulary
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
        try:
            job_details = self.clients["speech"].get_transcription_job(
                TranscriptionJobName=job_id
            )
        except Exception as exp:
            if "job couldn't be found" in str(exp):
                raise AsyncJobException(
                        reason = AsyncJobExceptionReason.DEPRECATED_JOB_ID
                    )
            raise ProviderException(str(exp))
        
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
                # diarization
                diarization_entries = []
                words_info = original_response["results"]["items"]
                speakers = original_response["results"].get("speaker_labels", {}).get("speakers", 0)

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
