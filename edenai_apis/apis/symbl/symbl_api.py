from io import BufferedReader
import json
import requests

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
    AsyncErrorResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)


class SymblApi(ProviderApi, Audio):
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
        self, file: BufferedReader, language: str,
        speakers : int
    ) -> AsyncLaunchJobResponseType:
        file.seek(0, 2)
        number_of_bytes = file.tell()
        file.seek(0)

        headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Length": str(number_of_bytes),
        }

        params = {
            "languageCode": language, 
            "enableSpeakerDiarization" : "true",
            "diarizationSpeakerCount" : speakers
            }

        response = requests.post(
            url="https://api.symbl.ai/v1/process/audio",
            headers=headers,
            data=file,
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

        if original_response["status"] == "completed":
            url = f"https://api.symbl.ai/v1/conversations/{conversation_id}/messages?sentiment=true"
            response = requests.get(url=url, headers=headers)
            if response.status_code != 200:
                return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                    error=response_status.text, provider_job_id=provider_job_id
                )

            original_response = response.json()
            diarization_entries = []
            speakers = set()

            text = " ".join(
                [message["text"] for message in original_response["messages"]]
            )

            for text_info in original_response["messages"]:
                speakers.add(text_info["from"]["name"])
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment= text_info["text"],
                        speaker= int(text_info["from"]["name"].split("Speaker")[1].strip()),
                        start_time= str(text_info["timeOffset"]),
                        end_time= str(text_info["timeOffset"] + text_info["duration"])
                    )
                )

            diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)

            standarized_response = SpeechToTextAsyncDataClass(text=text, diarization = diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )
        elif original_response["status"] == "failed":
            return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                error=response_status.text, provider_job_id=provider_job_id
            )
        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
