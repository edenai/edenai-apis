

from http import HTTPStatus
import json
from typing import Dict, Sequence
import uuid

import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text.spell_check.spell_check_dataclass import SpellCheckDataClass, SpellCheckItem, SuggestionItem
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class SaplingApi(ProviderInterface, TextInterface):
    provider_name = "sapling"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["key"]
        self.url = "https://api.sapling.ai/api/v1/spellcheck"


    def text__spell_check(
        self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        
        session_id = str(uuid.uuid4())
        payload = {
            "key": self.api_key,
            "text": text,
            "session_id": session_id,
            "multiple_edits": True
        }

        if language:
            payload.update({
                "lang": language
            })

        try:
            response = requests.post(self.url, json = payload)
        except Exception as excp:
            raise ProviderException(str(excp))
        
        original_response = response.json()
        
        if response.status_code > HTTPStatus.BAD_REQUEST:
            raise ProviderException(original_response)
                
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
                    suggestions.append(SuggestionItem(suggestion=word_candidate, score= None))

            if len(suggestions) == 0:
                suggestions.append(SuggestionItem(suggestion=edit["replacement"], score=None))
            items.append(
                SpellCheckItem(
                    text = checked_word,
                    offset= edit["start"] + text.index(edit["sentence"]),
                    length= end - start,
                    suggestions= suggestions,
                    type=None
                )
            )

        standardized_response = SpellCheckDataClass(text=text, items=items)

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )        
