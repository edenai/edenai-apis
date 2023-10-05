from http import HTTPStatus
import json
from typing import Dict, Sequence
import uuid

import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass
from edenai_apis.features.text.ai_detection.ai_detection_dataclass import (
    AiDetectionDataClass,
    AiDetectionItem,
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

from typing import Dict
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
    SegmentSentimentAnalysisDataClass,
)


from typing import cast


class SaplingApi(ProviderInterface, TextInterface):
    provider_name = "sapling"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["key"]
        self.url = "https://api.sapling.ai/api/v1/"

    def text__spell_check(
        self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        session_id = str(uuid.uuid4())
        payload = {
            "key": self.api_key,
            "text": text,
            "session_id": session_id,
            "multiple_edits": True,
        }

        if language:
            payload.update({"lang": language})

        try:
            response = requests.post(f"{self.url}spellcheck", json=payload)
        except Exception as excp:
            raise ProviderException(str(excp), code=500)

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal server error", code=response.status_code)

        try:
            original_response = response.json()
        except json.JSONDecodeError as excp:
            raise ProviderException("Response malformatted", code=response.status_code)

        if response.status_code > HTTPStatus.BAD_REQUEST:
            raise ProviderException(original_response, code=response.status_code)

        items: Sequence[SpellCheckItem] = []
        candidates = original_response.get("candidates", {})
        for edit in original_response.get("edits"):
            start = edit["start"]
            end = edit["end"]
            suggestions: Sequence[SuggestionItem] = []
            checked_word = edit["sentence"][start:end].strip()
            if checked_word in candidates:
                word_candidates = candidates[checked_word]
                for word_candidate in word_candidates:
                    suggestions.append(
                        SuggestionItem(suggestion=word_candidate, score=None)
                    )

            if len(suggestions) == 0:
                suggestions.append(
                    SuggestionItem(suggestion=edit["replacement"], score=None)
                )
            items.append(
                SpellCheckItem(
                    text=checked_word,
                    offset=edit["start"] + text.index(edit["sentence"]),
                    length=end - start,
                    suggestions=suggestions,
                    type=None,
                )
            )

        standardized_response = SpellCheckDataClass(text=text, items=items)

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        headers = {"Content-Type": "application/json"}
        payload = {"key": self.api_key, "text": text}

        try:
            response = requests.post(
                f"{self.url}sentiment", json=payload, headers=headers
            )
        except Exception as excp:
            raise ProviderException(str(excp), code=500)

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise ProviderException("Internal server error", code=response.status_code)

        response_json = response.json()

        if response.status_code > HTTPStatus.BAD_REQUEST:
            raise ProviderException(response_json, code=response.status_code)

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

        best_sentiment.update(
            {
                "general_sentiment": response_json["overall"][0][1],
                "general_sentiment_rate": response_json["overall"][0][0],
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

    def text__ai_detection(self, text: str) -> ResponseType[ChatDataClass]:
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

        original_response = response.json()

        if response.status_code > HTTPStatus.BAD_REQUEST:
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
            ai_score=original_response.get("score"), items=items
        )

        result = ResponseType[AiDetectionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
