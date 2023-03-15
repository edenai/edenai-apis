import requests
import uuid
from typing import Optional, List
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncResponseType,
    AsyncPendingResponseType,
)
from edenai_apis.features.audio.speech_to_text_async.speech_to_text_async_dataclass import (
    SpeechDiarization,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features import AudioInterface
from edenai_apis.utils.exception import ProviderException
import json
import urllib

class OpenaiAudioApi(AudioInterface):
    def audio__speech_to_text_async__launch_job(
        self, 
        file: str,
        language: str, 
        speakers: int, 
        profanity_filter: bool, 
        vocabulary: Optional[List[str]], 
        audio_attributes: tuple, 
        file_url: str = "") -> AsyncLaunchJobResponseType:
        data_job_id = {}
        webhook_token = self.api_settings['webhook_token']
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
        }
        url = 'https://api.openai.com/v1/audio/transcriptions'
        file_ = open(file, "rb")
        files = {
            'file':file_
        }
        payload = {
            'model':'whisper-1',
            'language':language
        }
        response = requests.post(url, data=payload, files=files, headers=headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)
        
        job_id = str(uuid.uuid4())
        data_job_id[job_id] = response.json()
        webhook_send =requests.post(
            url = f'https://webhook.site/{webhook_token}',
            data = json.dumps(data_job_id),
            
            headers = {'content-type':'application/json'})
        return AsyncLaunchJobResponseType(provider_job_id=job_id)
    
    def audio__speech_to_text_async__get_job_result(
        self,
        provider_job_id: str) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        
        if not provider_job_id:
            raise ProviderException("Job id None or empty!")
        
        # Get results from webhooks : 
        # List all webhook results
        token_id = self.api_settings['webhook_token']
        webhook_get_url = (
            f"https://webhook.site/token/{token_id}/requests"
            + f"?sorting=newest&query={urllib.parse.quote_plus('content:'+str(provider_job_id))}"
        )
        webhook_response = requests.get(webhook_get_url, headers={
            "api-key": self.api_settings['webhook_api_key']
        })

        response_status = webhook_response.status_code
        if response_status != 200 or len(
            webhook_response.json()["data"]
        ) == 0:
            original_response = None
        else:
            original_response= json.loads(webhook_response.json()["data"][0]["content"])
            original_response= original_response[provider_job_id]
            
        if original_response is None :
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )
            
        diarization = SpeechDiarization(total_speakers=0, entries= [])
        standardized_response = SpeechToTextAsyncDataClass(
            text = original_response.get('text'),
            diarization=diarization
            )
        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
            provider_job_id=provider_job_id,
        )