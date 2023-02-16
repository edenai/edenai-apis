from io import BufferedReader
from typing import List, Optional, Sequence
import requests

from edenai_apis.features import ProviderInterface, TextInterface, TranslationInterface
from edenai_apis.features.text import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    LanguageDetectionDataClass,
    InfosLanguageDetectionDataClass,
)
from edenai_apis.features.audio.speech_to_text_async import (
    SpeechToTextAsyncDataClass,
    SpeechDiarization
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    AsyncLaunchJobResponseType, ResponseType
)
from edenai_apis.utils.exception import ProviderException
from .config import get_domain_language_from_code


class NeuralSpaceApi(ProviderInterface, TextInterface, TranslationInterface):
    provider_name = "neuralspace"

    def __init__(self) -> None:
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api"]
        self.url = self.api_settings["url"]
        self.header = {
            "authorization": f"{self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        url = f"{self.url}ner/v1/entity"

        files = {"text": text, "language": language}

        response = requests.request("POST", url, json=files, headers=self.header)
        if response.status_code != 200:
            if not response.json().get("success"):
                raise ProviderException(response.json().get("message"))

        response = response.json()
        data = response["data"]

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        if len(data["entities"]) > 0:
            for entity in data["entities"]:
                text = entity["text"]
                if entity["type"] == 'time':
                    category = 'DATE'
                else:
                    category = entity["type"].upper()

                items.append(
                    InfosNamedEntityRecognitionDataClass(
                        entity=text,
                        importance=None,
                        category=category.replace('-',''),
                    )
                )

        standardized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=data, standardized_response=standardized_response
        )

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        url = f"{self.url}translation/v1/translate"

        files = {
            "text": text,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
        }

        response = requests.request("POST", url, json=files, headers=self.header)
        response = response.json()

        data = response["data"]

        if response["success"] is False:
            raise ProviderException(data["error"])

        standardized_response = AutomaticTranslationDataClass(
            text=data["translatedText"]
        )

        return ResponseType[AutomaticTranslationDataClass](
            original_response=data, standardized_response=standardized_response
        )

    def translation__language_detection(self, text) -> ResponseType[LanguageDetectionDataClass]:
        url = f"{self.url}language-detection/v1/detect"
        files = {"text": text}

        response = requests.request("POST", url, json=files, headers=self.header)

        original_response = response.json()
        if response.status_code != 200:
            raise ProviderException(message=original_response['error'], code=response.status_code)

        items: Sequence[InfosLanguageDetectionDataClass] = []
        for lang in original_response["data"]["detected_languages"]:
            confidence = float(lang["confidence"])
            if confidence > 0.1:
                items.append(
                    InfosLanguageDetectionDataClass(
                        language=lang["language"],
                        display_name=get_language_name_from_code(isocode=lang['language']),
                        confidence=confidence
                    )
                )
        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items)
        )


    def audio__speech_to_text_async__launch_job(
        self, 
        file: str, 
        language: str, 
        speakers: int,
        profanity_filter: bool, 
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:

        export_format, channels, frame_rate = audio_attributes

        url_file_upload = f"{self.url}file/upload"
        url_file_transcribe = f"{self.url}transcription/v1/file/transcribe"
        # first, upload file
        headers = {
            "Authorization" : f"{self.api_key}"
        }
        file_ = open(file, "rb")
        files = {"files" : file_}
        response = requests.post(
            url= url_file_upload,
            headers=headers,
            files= files
        )
        if response.status_code != 200:
            raise ProviderException("Failed to upload file for transcription", response.status_code)

        original_response = response.json()
        fileId = original_response.get('data').get('fileId')
    
        # then, call spech to text api
        language_domain = get_domain_language_from_code(language)
        payload= {
            "fileId": fileId
        }

        if language_domain:
            payload.update({
                "language": language_domain.get('language'),
                "domain" : language_domain.get('domain')
            })

        response = requests.post(
            url = url_file_transcribe,
            headers=headers,
            data= payload
        )
        original_response = response.json()
        if response.status_code != 201:
            raise ProviderException(original_response.get('data').get('error'))
        
        transcribeId = original_response.get('data').get('transcribeId')

        return AsyncLaunchJobResponseType(
            provider_job_id = transcribeId
        )

    
    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:

        url_transcribe = f"{self.url}transcription/v1/single/transcription?transcribeId={provider_job_id}"
        headers = {
            "Authorization" : f"{self.api_key}"
        }

        response= requests.get(
            url= url_transcribe,
            headers=headers
        )

        if response.status_code != 200:
            data = response.json().get("data") or {}
            # two message possible
            # ref: https://docs.neuralspace.ai/speech-to-text/transcribe-file
            error = response.json().get("message") or data.get("message")
            raise ProviderException(error)

        diarization = SpeechDiarization(total_speakers=0, entries= [])
        original_response = response.json()
        status = original_response.get('data').get('transcriptionStatus')
        if status != "Completed":
            return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                provider_job_id=provider_job_id
            )

        return AsyncResponseType[SpeechToTextAsyncDataClass](
            original_response = original_response,
            standardized_response = SpeechToTextAsyncDataClass(
                text = original_response.get('data').get('transcripts'),
                diarization= diarization
            ),
            provider_job_id = provider_job_id
        )
