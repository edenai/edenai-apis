import requests
import uuid
from typing import Optional, List
from edenai_apis.apis.amazon.helpers import check_webhook_result
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
        model :str,
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
        # Getting results from webhook.site

        wehbook_result, response_status = check_webhook_result(provider_job_id, self.api_settings)

        if response_status != 200:
            raise ProviderException(wehbook_result)
        
        result_object = next(filter(lambda response: provider_job_id in response["content"], wehbook_result), None) \
            if wehbook_result else None

        if not result_object or not result_object.get("content"):
            raise ProviderException("Provider returned an empty response")
                
        original_response = json.loads(result_object["content"]).get(provider_job_id, None)
        if original_response is None:
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