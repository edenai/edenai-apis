import json
from typing import Dict, List, Optional, Any

import requests

from edenai_apis.features import (
    AudioInterface,
    OcrInterface,
    ProviderInterface,
    TextInterface,
    TranslationInterface,
)
from edenai_apis.features.audio import (
    SpeechDiarization,
    SpeechDiarizationEntry,
    SpeechToTextAsyncDataClass,
)
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import OcrAsyncDataClass
from edenai_apis.features.text import (
    AnonymizationDataClass,
    InfosKeywordExtractionDataClass,
    InfosNamedEntityRecognitionDataClass,
    KeywordExtractionDataClass,
    NamedEntityRecognitionDataClass,
    SentimentAnalysisDataClass,
    SentimentEnum,
    SummarizeDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SegmentSentimentAnalysisDataClass,
)
from edenai_apis.features.translation import LanguageDetectionDataClass
from edenai_apis.features.translation.language_detection import (
    InfosLanguageDetectionDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)
from edenai_apis.utils.languages import get_code_from_language_name
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from .helpers import OneAIAsyncStatus


class OneaiApi(
    ProviderInterface, TextInterface, TranslationInterface, AudioInterface, OcrInterface
):
    provider_name = "oneai"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.oneai.com/api/v0/pipeline"
        self.header = {"api-key": self.api_key}

    def text__anonymization(
        self,
        text: str,
        language: str,
        model: Optional[str] = None,
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[AnonymizationDataClass]:
        data = json.dumps({"input": text, "steps": [{"skill": "anonymize"}]})

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        standardized_response = AnonymizationDataClass(
            result=original_response["output"][0]["text"]
        )

        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        payload = {
            "input": text,
            "input_type": "article",
            "content_type": "application/json",
            "output_type": "json",
            "multilingual": {"enabled": True},
            "steps": [{"skill": "keywords"}],
        }

        response = requests.post(url=self.url, headers=self.header, json=payload)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        items = []
        for item in original_response["output"][0]["labels"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=item["span_text"], importance=round(item["value"], 2)
                )
            )

        standardized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__named_entity_recognition(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        response = self.llm_client.named_entity_recognition(
            text=text, model=model, **kwargs
        )
        return response

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        data = json.dumps({"input": text, "steps": [{"skill": "sentiments"}]})

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        items = []
        general_sentiment = 0
        for item in original_response["output"][0]["labels"]:
            segment = item["span_text"]
            sentiment = (
                SentimentEnum.NEGATIVE
                if item["value"] == "NEG"
                else SentimentEnum.POSITIVE
            )
            general_sentiment += 1 if sentiment == SentimentEnum.POSITIVE else -1
            items.append(
                SegmentSentimentAnalysisDataClass(
                    segment=segment, sentiment=sentiment.value
                )
            )

        general_sentiment_text = SentimentEnum.NEUTRAL
        if general_sentiment < 0:
            general_sentiment_text = SentimentEnum.NEGATIVE
        elif general_sentiment > 0:
            general_sentiment = SentimentEnum.POSITIVE

        standardized_response = SentimentAnalysisDataClass(
            general_sentiment=general_sentiment_text.value, items=items
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[SummarizeDataClass]:
        data = json.dumps({"input": text, "steps": [{"skill": "summarize"}]})

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        text = original_response["output"][0]["text"]

        standardized_response = SummarizeDataClass(result=text)

        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def translation__language_detection(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:
        data = json.dumps(
            {
                "input": text,
                "steps": [{"skill": "detect-language"}],
                "multilingual": True,
            }
        )

        response = requests.post(url=self.url, headers=self.header, data=data)
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        items = []
        for item in original_response["output"][0]["labels"]:
            items.append(
                InfosLanguageDetectionDataClass(
                    language=get_code_from_language_name(name=item["value"]),
                    display_name=item["value"],
                )
            )

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items),
        )

    def audio__speech_to_text_async__launch_job(
        self,
        file: str,
        language: str,
        speakers: int,
        profanity_filter: bool,
        vocabulary: Optional[List[str]],
        audio_attributes: tuple,
        model: Optional[str] = None,
        file_url: str = "",
        provider_params: Optional[dict] = None,
        **kwargs,
    ) -> AsyncLaunchJobResponseType:
        provider_params = provider_params or {}
        export_format, channels, frame_rate = audio_attributes

        data = {
            "input_type": "conversation",
            "content_type": "audio/" + export_format,
            "steps": [
                {
                    "skill": "transcribe",
                    "params": {"speaker_detection": True, "engine": "whisper"},
                }
            ],
            "multilingual": True,
            **provider_params,
        }

        with open(file, "rb") as file_:
            response = requests.post(
                url=f"{self.url}/async/file?pipeline={json.dumps(data)}",
                headers=self.header,
                data=file_.read(),
            )
        original_response = response.json()

        if response.status_code != 200:
            raise ProviderException(
                message=original_response["message"], code=response.status_code
            )

        return AsyncLaunchJobResponseType(provider_job_id=original_response["task_id"])

    def audio__speech_to_text_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        response = requests.get(
            url=f"{self.url}/async/tasks/{provider_job_id}", headers=self.header
        )

        original_response = response.json()

        if response.status_code == 200:
            if original_response["status"] == OneAIAsyncStatus.COMPLETED.value:
                final_text = ""
                phrase = original_response["result"]["input_text"].split("\n\n")
                for item in phrase:
                    if item != "":
                        *options, text = item.split("\n")
                        final_text += f"{text} "

                diarization_entries = []
                speakers = set()
                words_info = original_response["result"]["output"][0]["labels"]

                for word_info in words_info:
                    if word_info.get("speaker"):
                        speakers.add(word_info["speaker"])
                        diarization_entries.append(
                            SpeechDiarizationEntry(
                                segment=word_info["span_text"],
                                start_time=word_info["timestamp"],
                                end_time=word_info["timestamp_end"],
                                speaker=int(word_info["speaker"].split("speaker")[1]),
                            )
                        )
                diarization = SpeechDiarization(
                    total_speakers=len(speakers), entries=diarization_entries
                )
                standardized_response = SpeechToTextAsyncDataClass(
                    text=final_text.strip(), diarization=diarization
                )
                return AsyncResponseType[SpeechToTextAsyncDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                    provider_job_id=provider_job_id,
                )
            elif original_response["status"] == OneAIAsyncStatus.RUNNING.value:
                return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
                    provider_job_id=provider_job_id
                )
            else:
                raise ProviderException(original_response)
        else:
            if original_response.get("status") == "NOT_FOUND":
                raise AsyncJobException(
                    reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID
                )
            raise ProviderException(original_response)

    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        params = {
            "input_type": "article",
            "content_type": "text/pdf",
            "steps": [{"skill": "pdf-extract-text"}],
        }

        if file_url:
            params["input"] = file_url
            file_param = None
        else:
            with open(file, "rb") as _file:
                file_param = _file.read()

        response = requests.post(
            f"{self.url}/async/file",
            params={"pipeline": json.dumps(params)},
            headers=self.header,
            data=file_param,
        )

        if not response.ok:
            raise ProviderException(
                message=response.json()["message"], code=response.status_code
            )

        data = response.json()

        return AsyncLaunchJobResponseType(provider_job_id=data["task_id"])

    def ocr__ocr_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrAsyncDataClass]:
        response = requests.get(
            url=f"{self.url}/async/tasks/{provider_job_id}", headers=self.header
        )
        status_code = response.status_code
        original_response = response.json()
        status = original_response["status"]

        if status in (OneAIAsyncStatus.RUNNING.value, OneAIAsyncStatus.QUEUED.value):
            return AsyncPendingResponseType(provider_job_id=provider_job_id)
        elif status == OneAIAsyncStatus.COMPLETED.value:
            standardized_response = OcrAsyncDataClass(
                raw_text=original_response["result"]["output"][0]["text"]
            )
            return AsyncResponseType(
                provider_job_id=provider_job_id,
                original_response=original_response,
                standardized_response=standardized_response,
            )
        elif status == OneAIAsyncStatus.FAILED.value:
            raise ProviderException(original_response, code=status_code)
        elif status == OneAIAsyncStatus.NOT_FOUND.value:
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID, code=status_code
            )
        else:
            raise ProviderException(original_response, code=status_code)
