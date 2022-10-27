from io import BufferedReader
import requests

from edenai_apis.features import ProviderApi, Audio
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
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


class RevAIApi(ProviderApi, Audio):
    provider_name = "revai"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.key = self.api_settings["revai_key"]

    def audio__speech_to_text_async__launch_job(
        self, file: BufferedReader, language: str
    ) -> AsyncLaunchJobResponseType:

        response = requests.post(
            url="https://ec1.api.rev.ai/speechtotext/v1/jobs",
            headers={"Authorization": f"Bearer {self.key}"},
            data={"options": "{}", "language": f"{language}"},
            files=[("media", ("audio_file", file))],
        )
        original_response = response.json()

        if response.status_code != 200:
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
        return AsyncLaunchJobResponseType(provider_job_id=original_response["id"])

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        headers = {"Authorization": f"Bearer {self.key}"}
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
        else:
            status = original_response["status"]
            if status == "transcribed":
                response = requests.get(
                    url=f"https://ec1.api.rev.ai/speechtotext/v1/jobs/{provider_job_id}/transcript",
                    headers=headers,
                )
                if response.status_code != 200:
                    raise ProviderException(
                        message=f"{original_response.get('title','')}: {original_response.get('details','')}",
                        code=response.status_code,
                    )
                else:
                    original_response = response.json()
                    text = ""
                    for monologue in original_response["monologues"]:
                        text += "".join(
                            [element["value"] for element in monologue["elements"]]
                        )
                    standarized_response = SpeechToTextAsyncDataClass(text=text)
                    return AsyncResponseType[SpeechToTextAsyncDataClass](
                        original_response=original_response,
                        standarized_response=standarized_response,
                        provider_job_id=provider_job_id,
                    )
            elif status == "failed":
                return AsyncErrorResponseType[SpeechToTextAsyncDataClass](
                    error=original_response["failure_detail"],
                    provider_job_id=provider_job_id,
                )
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
