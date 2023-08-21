from typing import Sequence
from edenai_apis.apis.ibm.ibm_helpers import handle_ibm_call

from edenai_apis.features.text import (
    ExtractedTopic,
    InfosKeywordExtractionDataClass,
    InfosNamedEntityRecognitionDataClass,
    InfosSyntaxAnalysisDataClass,
    KeywordExtractionDataClass,
    NamedEntityRecognitionDataClass,
    SegmentSentimentAnalysisDataClass,
    SentimentAnalysisDataClass,
    SyntaxAnalysisDataClass,
    TopicExtractionDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.exception import LanguageException, ProviderException
from edenai_apis.utils.types import ResponseType
from ibm_watson.natural_language_understanding_v1 import (
    CategoriesOptions,
    EntitiesOptions,
    Features,
    KeywordsOptions,
    SentimentOptions,
    SyntaxOptions,
    SyntaxOptionsTokens,
)
from watson_developer_cloud.watson_service import WatsonApiException

from .config import tags


class IbmTextApi(TextInterface):
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(sentiment=SentimentOptions())
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)

        # Create output object
        items: Sequence[SegmentSentimentAnalysisDataClass] = []
        standarize = SentimentAnalysisDataClass(
            general_sentiment=response["sentiment"]["document"]["label"],
            general_sentiment_rate=float(
                abs(response["sentiment"]["document"]["score"])
            ),
            items=items,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standardized_response=standarize
        )

    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(keywords=KeywordsOptions(emotion=True, sentiment=True))
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)
        
        # Analysing response
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in response["keywords"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=key_phrase["text"], importance=key_phrase["relevance"]
                )
            )

        standardized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(entities=EntitiesOptions(sentiment=True, mentions=True, emotion=True))
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)
        
        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        for ent in response["entities"]:
            category = ent["type"].upper()
            if category == "JOBTITLE":
                category = "PERSONTYPE"
            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=ent["text"],
                    importance=ent["relevance"],
                    category=category,
                )
            )

        standardized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__syntax_analysis(
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(syntax=SyntaxOptions(sentences=True,
                            tokens=SyntaxOptionsTokens(lemma=True, part_of_speech=True)))
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        response = handle_ibm_call(request.get_result)
        
        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Getting syntax detected of word and its score of confidence
        for keyword in response["syntax"]["tokens"]:
            tag_ = tags[keyword["part_of_speech"]]
            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=keyword["text"],
                    importance=None,
                    others=None,
                    tag=tag_,
                    lemma=keyword.get("lemma"),
                )
            )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        return ResponseType[SyntaxAnalysisDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        payload = {
            "text": text,
            "language": language,
            "features": Features(categories=CategoriesOptions())
        }
        request = handle_ibm_call(self.clients["text"].analyze, **payload)
        original_response = handle_ibm_call(request.get_result)
        
        categories: Sequence[ExtractedTopic] = []
        for category in original_response.get("categories"):
            categories.append(
                ExtractedTopic(
                    category=category.get("label"), importance=category.get("score")
                )
            )

        standardized_response = TopicExtractionDataClass(items=categories)
        result = ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result
