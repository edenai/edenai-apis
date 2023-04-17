from io import BufferedReader
import json
import requests
import re
import os

from edenai_apis.features import ProviderInterface, AudioInterface
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import AsyncJobException, AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.files import FileWrapper
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)


class SymblApi(ProviderInterface, AudioInterface):
    provider_name = "symbl"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.app_id = self.api_settings["app_id"]
        self.app_secret = self.api_settings["app_secret"]
        self._get_access_token()

    def _get_access_token(self) -> None:
        """
        Need to generate a token with app_id & app_secret
        the access Token will last for 24h only.
        If we call the endpoint while token is still active,
        it will return the active token, otherwise it creates a new one.
        Ref: https://docs.symbl.ai/docs/developer-tools/authentication/
        """

        payload = {
            "type": "application",
            "appId": self.app_id,
            "appSecret": self.app_secret,
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            "https://api.symbl.ai/oauth2/token:generate",
            headers=headers,
            data=json.dumps(payload),
        )
        self.access_token = response.json()["accessToken"]



    def audio__speech_to_text_async__launch_job(
        self, 
        file: str, 
        language: str,
        speakers : int, 
        profanity_filter: bool,
        vocabulary: list, 
        audio_attributes: tuple,
        model : str = None,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes

        number_of_bytes = os.stat(file).st_size

        headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Length": str(number_of_bytes),
        }

        params = {
            "enableSpeakerDiarization" : "true",
            "diarizationSpeakerCount" : speakers
            }
        if language:
            params.update({"languageCode": language})
        if vocabulary:
            if len(vocabulary) == 1:
                vocabulary.append(vocabulary[0])
            params.update({
                "customVocabulary": vocabulary
            })

        file_ = open(file, "rb")
        response = requests.post(
            url="https://api.symbl.ai/v1/process/audio",
            headers=headers,
            data=file_,
            params=params,
        )

        if response.status_code != 201:
            raise ProviderException(
                f"Call to Symbl failed.\nResponse Status: {response.status_code}.\n"
                + f"Response Content: {response.content}"
            )

        original_response = response.json()
        job_id = (
            original_response["jobId"] + "###" + original_response["conversationId"]
        )

        return AsyncLaunchJobResponseType(
            provider_job_id=job_id
        )

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        job_id, conversation_id = provider_job_id.split("###")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

        url_status = f"https://api.symbl.ai/v1/job/{job_id}"

        response_status = requests.get(url=url_status, headers=headers)
        original_response = response_status.json()

        if not original_response.get("status"):
            if isinstance(original_response.get("message"), str):
                error_message = original_response.get("message")
                if all(fraction in error_message for fraction in ["Job with", "not found"]):
                    raise AsyncJobException(
                        reason = AsyncJobExceptionReason.DEPRECATED_JOB_ID
                    )
            raise ProviderException(original_response.get("message"))

        if original_response["status"] == "completed":
            url = f"https://api.symbl.ai/v1/conversations/{conversation_id}/messages?sentiment=true&verbose=true"
            response = requests.get(url=url, headers=headers)
            if response.status_code != 200:
                raise ProviderException(response_status.text)

            original_response = response.json()
            print(json.dumps(original_response, indent=2))
            diarization_entries = []
            speakers = set()

            text = " ".join([message["text"] for message in original_response["messages"]])

            for text_info in original_response["messages"]:
                words_info = text_info["words"]
                for word_info in words_info:
                    speakers.add(word_info.get("speakerTag",1))
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            segment= word_info["word"],
                            speaker= word_info.get("speakerTag", 1),
                            start_time= str(word_info["timeOffset"]),
                            end_time= str(word_info["timeOffset"] + word_info["duration"]),
                            confidence= word_info.get("score")
                        )
                    )

            diarization = SpeechDiarization(
                total_speakers=len(speakers), entries=diarization_entries
            )

            standardized_response = SpeechToTextAsyncDataClass(text=text, diarization=diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
                provider_job_id=provider_job_id,
            )
        elif original_response["status"] == "failed":
            raise ProviderException(response_status.text)
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
