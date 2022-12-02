from io import BufferedReader
import requests
import uuid
from time import time
import json
import boto3
from botocore.errorfactory import ClientError

from edenai_apis.features import ProviderApi, Audio
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
from apis.amazon.config import clients, storage_clients


class RevAIApi(ProviderApi, Audio):
    provider_name = "revai"

    bucket_name = "716b4c0d6f124c75be2049d329cd0783"
    bucket_region = "eu-west-1"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.key = self.api_settings["revai_key"]


    def _create_vocabulary(self, list_vocabs: list):
        vocab_name = str(uuid.uuid4())
        response = requests.post(
            url="https://api.rev.ai/speechtotext/v1/vocabularies",
            headers={"Authorization": f"Bearer {self.key}",},
            data= {
                "custom_vocabularies" : [{
                    "phrases" : list_vocabs
                }]
            }
        )
        original_response = response.json()
        if response.status_code != 200:
            print(original_response)
            message = f"{original_response.get('title','')}: {original_response.get('details','')}"
            raise ProviderException(
                message=message,
                code=response.status_code,
            )
        vocab_name = original_response["id"]

        return vocab_name
    

    def _launch_transcribe(
        self, filename:str, language:str,
        profanity_filter: bool, vocab_name:str=None, 
        initiate_vocab:bool= False):

        file_url = f"https://{self.bucket_name}.s3.{self.bucket_region}.amazonaws.com/{filename}"

        config = {
            "filter_profanity": profanity_filter,
        }

        if language:
            config.update({
                "language": language,
            })

        if vocab_name:
            config.update({
                "custom_vocabulary_id" : vocab_name
            })
            if initiate_vocab:
                config["checked"]= False
                filename = f"{vocab_name}_settings.txt"
                storage_clients["speech"].meta.client.put_object(
                    Bucket= self.bucket_name, 
                    Body=json.dumps(config).encode(), 
                    Key=filename
                )
                return 

        data_config = {
                "options": json.dumps(config),
                "source_config" : {
                    "url" : file_url
                }
            }
        print(data_config)
        response = requests.post(
            url="https://ec1.api.rev.ai/speechtotext/v1/jobs",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-type": "application/json", 
                },
            data= json.dumps(data_config),
            # files=[("media", ("audio_file", file))],
        )
        original_response = response.json()
    
        if response.status_code != 200:
            print(response.json())
            parameters = original_response.get('parameters')
            for key, value in parameters.items():
                if "filter_profanity" in key:
                    raise ProviderException(f"{key}: {value[0]} Use 'en' language for profanity filter")
            message = f"{original_response.get('title','')}: {original_response.get('details','')}"
            if message and message[0] == ":":
                if len(message) > 2:
                    message = message[2:]
                else:
                    message = "An error has occurred..."
            raise ProviderException(
                message=message,
                code=response.status_code,
            )
        return original_response["id"]


    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str,
        speakers : int, profanity_filter: bool, vocabulary: list
    ) -> AsyncLaunchJobResponseType:

        # upload file to amazon S3
        file_name = str(int(time())) + "_" + str(file.name.split("/")[-1])
        storage_clients["speech"].meta.client.upload_fileobj(
            Fileobj = file, 
            Bucket = self.bucket_name, 
            Key= file_name 
        )

        if vocabulary:
            vocab_name = self._create_vocabulary(vocabulary)
            self._launch_transcribe(file_name, language, profanity_filter, vocab_name, True)
            return AsyncLaunchJobResponseType(provider_job_id=f"{vocab_name}EdenAI{file_name}")


        provider_job_id = self._launch_transcribe(file_name, 
            language, profanity_filter)
        provider_job_id = f"{provider_job_id}EdenAI{file_name}" # file_name to delete file

        return AsyncLaunchJobResponseType(provider_job_id=provider_job_id)


    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        provider_job_id, file_name = provider_job_id.split("EdenAI")
        headers = {"Authorization": f"Bearer {self.key}"}
        # check if custom vocabulary
        try: # check if job id of vocabulary or not
            config_content = storage_clients["speech"].meta.client.get_object(
                Bucket = self.bucket_name,
                Key= f"{provider_job_id}_settings.txt"
                )
            config = json.loads(config_content['Body'].read().decode('utf-8'))
            #check if vocab finished
            if job_id:= not config.get("provider_job_id"): # check if transcribe have been launched
                response = requests.get(
                    url=f"https://ec1.api.rev.ai/speechtotext/v1/vocabularies/{provider_job_id}",
                    headers=headers,
                )
                original_response = response.json()
                if response.status_code != 200:
                    raise ProviderException(
                        message=f"{original_response.get('title','')}: {original_response.get('details','')}",
                        code=response.status_code,
                    )
                status = original_response["status"]
                if status == "in_progress":
                    return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                        provider_job_id=provider_job_id
                    )
                if status == "failed":
                    return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                        error=original_response["failure_detail"],
                        provider_job_id=provider_job_id,
                    )
                provider_job = self._launch_transcribe(
                    config["source_config"]["url"].split("/")[-1],
                    config.get("language"),
                    config["filter_profanity"],
                    provider_job_id
                )
                config["provider_job_id"] = provider_job
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                    provider_job_id=provider_job_id
                )
            provider_job_id = job_id
            
        except ClientError as exc:
            pass

        response = requests.get(
            url=f"https://ec1.api.rev.ai/speechtotext/v1/jobs/{provider_job_id}",
            headers=headers,
        )
        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(
                message=f"{original_response.get('title','')}: {original_response.get('details','')}",
                code=response.status_code,
            )

        status = original_response["status"]
        if status == "transcribed":
            # delete audio file
            storage_clients["speech"].meta.client.delete_object(
                Bucket= self.bucket_name, 
                Key= file_name
            )
            response = requests.get(
                url=f"https://ec1.api.rev.ai/speechtotext/v1/jobs/{provider_job_id}/transcript",
                headers=headers,
            )
            if response.status_code != 200:
                raise ProviderException(
                    message=f"{original_response.get('title','')}: {original_response.get('details','')}",
                    code=response.status_code,
                )

            diarization_entries = []
            speakers = set()

            original_response = response.json()
            text = ""
            for monologue in original_response["monologues"]:
                text += "".join(
                    [element["value"] for element in monologue["elements"]]
                )
                speakers.add(monologue["speaker"])
                for word_info in monologue["elements"]:
                    if word_info["type"] == "text":
                        diarization_entries.append(
                            SpeechDiarizationEntry(
                                speaker= monologue["speaker"] + 1,
                                segment= word_info["value"],
                                start_time= str(word_info["ts"]),
                                end_time= str(word_info["end_ts"]),
                                confidence= word_info["confidence"]
                            )
                        )

            diarization = SpeechDiarization(total_speakers= len(speakers), entries= diarization_entries)

            standarized_response = SpeechToTextAsyncDataClass(text=text, diarization= diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )
        elif status == "failed":
            error = original_response.get("failure_detail")
            raise ProviderException(error)
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
