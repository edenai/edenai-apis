import json
import uuid
from http import HTTPStatus
from typing import Dict, List, Optional, Any
from typing import cast

import requests

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass,
    AiDetectionItem,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
    SegmentSentimentAnalysisDataClass,
)
from edenai_apis.features.text.spell_check.spell_check_dataclass import (
    SpellCheckDataClass,
    SpellCheckItem,
    SuggestionItem,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class SaplingApi(ProviderInterface, TextInterface):
    provider_name = "sapling"

    def __init__(self, api_keys: Optional[Dict] = None) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["key"]
        self.url = "https://api.sapling.ai/api/v1/"

    @staticmethod
    def _check_error(response: requests.Response) -> None:
        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal server error", code=response.status_code)

        try:
            response_json = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(response.text, code=response.status_code) from exc

        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise ProviderException(
                response_json.get("msg", response.text), code=response.status_code
            )

    def text__spell_check(
        self, text: str, language: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SpellCheckDataClass]:
        session_id = str(uuid.uuid4())
        payload = {
            "key": self.api_key,
            "text": text,
            "session_id": session_id,
            "multiple_edits": True,
        }

        if language is not None:
            payload["lang"] = language

        response = requests.post(f"{self.url}spellcheck", json=payload)
        SaplingApi._check_error(response)
        original_response = response.json()

        items: List[SpellCheckItem] = []
        candidates = original_response.get("candidates", {})
        edits = original_response.get("edits", [])

        def extract_item(misspelled_word: Dict[str, Any]) -> SpellCheckItem:
            end, _, replacement, __, sentence_start, start = misspelled_word.values()
            offset = start + sentence_start
            length = end - start
            checked_word = text[offset : offset + length]
            list_of_suggestions = candidates.get(checked_word, [replacement])

            suggestions: List[SuggestionItem] = [
                SuggestionItem(suggestion=suggestion, score=None)
                for suggestion in list_of_suggestions
            ]
            return SpellCheckItem(
                text=checked_word,
                offset=offset,
                length=length,
                suggestions=suggestions,
                type=None,
            )

        for edit in edits:
            extracted_item = extract_item(edit)
            items.append(extracted_item)

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        headers = {"Content-Type": "application/json"}
        payload = {"key": self.api_key, "text": text}

        response = requests.post(f"{self.url}sentiment", json=payload, headers=headers)

        SaplingApi._check_error(response)
        response_json = response.json()

        best_sentiment = {
            "general_sentiment": None,
            "general_sentiment_rate": 0,
            "items": [],
        }

        for i in range(len(response_json.get("sents") or [])):
            sentence = response_json["sents"][i]
            sentiment = response_json["results"][i][0][1]
            sentiment_rate = response_json["results"][i][0][0]

            if sentiment != "Mixed":
                segment_sentiment = SegmentSentimentAnalysisDataClass(
                    segment=sentence, sentiment=sentiment, sentiment_rate=sentiment_rate
                )
                best_sentiment["items"].append(segment_sentiment)
        overall = response_json.get("overall", [])
        best_sentiment.update(
            {
                "general_sentiment": overall[0][1] if overall else None,
                "general_sentiment_rate": overall[0][0] if overall else 0,
            }
        )

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_sentiment["general_sentiment"],
            general_sentiment_rate=best_sentiment["general_sentiment_rate"],
            items=best_sentiment["items"],
        )

        return cast(
            ResponseType[SentimentAnalysisDataClass],
            ResponseType(
                original_response=response_json, standardized_response=standarize
            ),
        )

    def text__ai_detection(
        self, text: str, provider_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ResponseType[AiDetectionDataClass]:
        payload = {
            "key": self.api_key,
            "text": text,
        }

        try:
            response = requests.post(f"{self.url}aidetect", json=payload)
        except Exception as excp:
            raise ProviderException(str(excp), code=500)

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal server error", code=response.status_code)

        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(response.text, code=response.status_code) from exc

        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise ProviderException(original_response, code=response.status_code)

        items = []
        for sentence_score in original_response.get("sentence_scores", {}):
            ai_score = sentence_score["score"]
            items.append(
                AiDetectionItem(
                    text=sentence_score["sentence"],
                    prediction=AiDetectionItem.set_label_based_on_score(ai_score),
                    ai_score=ai_score,
                )
            )

        standardized_response = AiDetectionDataClass(
            ai_score=original_response.get("score") or 0, items=items
        )

        result = ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
