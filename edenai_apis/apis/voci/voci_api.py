from io import BufferedReader
import requests
from edenai_apis.features.audio.speech_to_text_async import (
    SpeechToTextAsyncDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.exception import ProviderException
from edenai_apis.features import Audio
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncErrorResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
)
import json


class VociApi(ProviderApi, Audio):
    provider_name = "voci"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.key = self.api_settings["voci_key"]

    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str,
        speakers : int
    ) -> AsyncLaunchJobResponseType:
        wav_file = wav_converter(file, channels=1)[0]

        response = requests.post(
            url="https://vcloud.vocitec.com/transcribe",
            data={
                "output": "json",
                "token": self.key,
                "model": f"{language.lower()}:callcenter",
                "filetype": "audio/x-wav",
                "emotion": "true",
                "gender": "true",
                "diarize": "true"
            },
            files=[("file", wav_file)],
        )
        if response.status_code != 200:
            raise ProviderException(
                f"Call to Voci failed.\nResponse Status: {response.status_code}.\n"
                + f"Response Content: {response.content}"
            )
        else:
            original_response = response.json()
            job_id = original_response["requestid"]
            return AsyncLaunchJobResponseType(provider_job_id=job_id)

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        payload = {"token": self.key, "requestid": provider_job_id}
        response = requests.get(
            url="https://vcloud.vocitec.com/transcribe/result", params=payload
        )
        if response.status_code == 200:
            url = response.json()
            response_text = requests.get(url=url)

            if response_text.status_code != 200:
                raise ProviderException(
                    f"Call to Voci failed.\nResponse Status: {response.status_code}.\n"
                    + f"Response Content: {response.content}"
                )

            original_response = response_text.json()

            diarization_entries = []
            speakers = original_response["nchannels"] + 1
            
            text = ""
            for utterance in original_response.get("utterances"):
                for event in utterance.get("events"):
                    text += event.get("word") + " "
                    
                    diarization_entries.append(
                        SpeechDiarizationEntry(
                            segment= event["word"],
                            start_time= str(event["start"]),
                            end_time= str(event["end"]),
                            confidence= event["confidence"],
                            speaker= utterance["metadata"]["channel"] + 1
                        )
                    )
            diarization = SpeechDiarization(total_speakers=speakers, entries=diarization_entries)
            standarized_response = SpeechToTextAsyncDataClass(text=text, diarization=diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response=original_response,
                standarized_response=standarized_response,
                provider_job_id=provider_job_id,
            )
        elif response.status_code == 202:
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
        else:
            error = response.status_code
            return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                error=error, provider_job_id=provider_job_id
            )
