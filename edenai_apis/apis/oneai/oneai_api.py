from pprint import pprint
from io import BufferedReader
import json
from typing import Optional

import requests
from edenai_apis.features import ProviderApi, Text, Translation, Audio
from edenai_apis.features.audio import SpeechToTextAsyncDataClass
from edenai_apis.features.text import (
    AnonymizationDataClass,
    KeywordExtractionDataClass,
    NamedEntityRecognitionDataClass,
    InfosNamedEntityRecognitionDataClass,
    InfosKeywordExtractionDataClass,
    SentimentAnalysisDataClass,
    SummarizeDataClass,
    Items,
)
from edenai_apis.features.translation import (
    LanguageDetectionDataClass,
)
from edenai_apis.features.translation.language_detection.language_detection_dataclass import InfosLanguageDetectionDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import wav_converter
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncErrorResponseType,
    AsyncResponseType,
    ResponseType
)

class StatusEnum(enumerate):
    SUCCESS = 'COMPLETED'
    RUNNING = 'RUNNING'
    FAILED = 'FAILED'
class OneaiApi(ProviderApi, Text, Translation, Audio):
    provider_name = 'oneai'

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings['api_key']
        self.url = self.api_settings['url']
        self.header = {
            "api-key": self.api_key,
            "accept": "application/json",
            "Content-Type": "application/json",
        }

    
    def text__anonymization(self, text: str, language: str) -> ResponseType[AnonymizationDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "anonymize"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)
        
        standarized_response = AnonymizationDataClass(result=original_response['output'][0]['text'])

        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )


    def text__keyword_extraction(self, language: str, text: str) -> ResponseType[KeywordExtractionDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "keywords"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        items = []
        for item in original_response['output'][0]['labels']:
            items.append(InfosKeywordExtractionDataClass(keyword=item['span_text'], importance=item['value']))

        standarized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def text__named_entity_recognition(self, language: str, text: str) -> ResponseType[NamedEntityRecognitionDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "names"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        items = []
        for item in original_response['output'][0]['labels']:
            items.append(InfosNamedEntityRecognitionDataClass(entity=item['value'], category=item['name']))

        standarized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def text__sentiment_analysis(self, language: str, text: str) -> ResponseType[SentimentAnalysisDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "sentiments"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        items = []
        for item in original_response['output'][0]['labels']:
            items.append(Items(sentiment='negative' if item['value'] == 'NEG' else 'positive'))

        standarized_response = SentimentAnalysisDataClass(items=items)

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def text__summarize(self, text: str, output_sentences: int, language: str, model: Optional[str]) -> ResponseType[SummarizeDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "summarize"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        text = original_response['output'][0]['text']

        standarized_response = SummarizeDataClass(result=text)

        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def translation__language_detection(self, text: str) -> ResponseType[LanguageDetectionDataClass]:
        data = json.dumps({
            "input": text,
            "steps": [
                {
                    "skill": "detect-language"
                }
            ]
        })

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        items = []
        for item in original_response['output'][0]['labels']:
            items.append(InfosLanguageDetectionDataClass(language=item['value']))

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )

    def audio__speech_to_text_async__launch_job(self, file: BufferedReader, language: str) -> AsyncLaunchJobResponseType:
        wav_file = wav_converter(file, frame_rate=16000, channels=1)[0]

        data = json.dumps({
            "input_type": 'conversation',
            "content_type": "audio/wav",
            "steps": [
                {
                    "skill": "transcribe",
                    "params": {
                        "speaker_detection": True
                    }
                }  
            ]
        })

        response = requests.post(url=f"{self.url}/async/file?pipeline={data}", headers=self.header, data=wav_file.read())
        
        print(response)
        
        original_response = response.json()
        print(original_response)

        if response.status_code != 200:
            raise ProviderException(message=original_response['message'], code=response.status_code)

        return AsyncLaunchJobResponseType(
            provider_job_id=original_response['task_id']
        )        


    def audio__speech_to_text_async__get_job_result(self, provider_job_id: str) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        response = requests.get(url=f"{self.url}/async/tasks/{provider_job_id}", headers=self.header)

        original_response = response.json()

        if response.status_code == 200:
            pprint(original_response)
            if original_response['status'] == StatusEnum.SUCCESS:
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response=original_response,
                    standarized_response=SpeechToTextAsyncDataClass(text=original_response['result']['input_text']),
                    provider_job_id=provider_job_id
                )
            elif original_response['status'] == StatusEnum.RUNNING:
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](provider_job_id=provider_job_id)
            else:
                return AsyncErrorResponseType(provider_job_id=provider_job_id, error=original_response)
        else:
            return AsyncErrorResponseType(provider_job_id=provider_job_id, error=original_response)