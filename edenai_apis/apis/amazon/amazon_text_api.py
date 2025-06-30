from typing import List, Literal, Optional, Sequence, Union, Dict, Any
import json

from edenai_apis.apis.amazon.helpers import handle_amazon_call
from edenai_apis.features.text import ChatDataClass, GenerationDataClass
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
    AnonymizationEntity,
)
from edenai_apis.features.text.anonymization.category import CategoryType
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatDataClass
from edenai_apis.features.text.entity_sentiment.entities import Entities
from edenai_apis.features.text.entity_sentiment.entity_sentiment_dataclass import (
    Entity,
    EntitySentimentDataClass,
)
from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    InfosKeywordExtractionDataClass,
    KeywordExtractionDataClass,
)
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.syntax_analysis.syntax_analysis_dataclass import (
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.utils.types import ResponseType
from .config import tags


class AmazonTextApi(TextInterface):
    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        # Getting response
        payload = {"Text": text, "LanguageCode": language}
        response = handle_amazon_call(self.clients["text"].detect_sentiment, **payload)

        # Analysing response

        best_sentiment = {
            "general_sentiment": None,
            "general_sentiment_rate": 0,
            "items": [],
        }

        for key in response["SentimentScore"]:
            if key == "Mixed":
                continue

            if (
                best_sentiment["general_sentiment_rate"]
                <= response["SentimentScore"][key]
            ):
                best_sentiment["general_sentiment"] = key
                best_sentiment["general_sentiment_rate"] = response["SentimentScore"][
                    key
                ]

        standarize = SentimentAnalysisDataClass(
            general_sentiment=best_sentiment["general_sentiment"],
            general_sentiment_rate=best_sentiment["general_sentiment_rate"],
            items=[],
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standardized_response=standarize
        )

    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        # Getting response
        payload = {"Text": text, "LanguageCode": language}
        response = handle_amazon_call(
            self.clients["text"].detect_key_phrases, **payload
        )

        # Analysing response
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in response["KeyPhrases"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=key_phrase["Text"], importance=key_phrase["Score"]
                )
            )

        standardized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__named_entity_recognition(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        # Getting response
        payload = {"Text": text, "LanguageCode": language}
        response = handle_amazon_call(self.clients["text"].detect_entities, **payload)

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []
        for ent in response["Entities"]:
            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=ent["Text"],
                    importance=ent["Score"],
                    category=ent["Type"],
                )
            )

        standardized = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=response, standardized_response=standardized
        )

    def text__syntax_analysis(
        self, language: str, text: str, **kwargs
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        # Getting response
        payload = {"Text": text, "LanguageCode": language}
        response = handle_amazon_call(self.clients["text"].detect_syntax, **payload)

        # Create output TextSyntaxAnalysis object

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Analysing response
        #
        # Getting syntax detected of word and its score of confidence
        for ent in response["SyntaxTokens"]:
            tag = tags[ent["PartOfSpeech"]["Tag"]]
            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=ent["Text"],
                    importance=ent["PartOfSpeech"]["Score"],
                    tag=tag,
                    lemma=None,
                    others=None,
                )
            )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        return ResponseType[SyntaxAnalysisDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__anonymization(
        self,
        text: str,
        language: str,
        model: Optional[str] = None,
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[AnonymizationDataClass]:
        payload = {"Text": text, "LanguageCode": language}
        res = handle_amazon_call(self.clients["text"].detect_pii_entities, **payload)

        last_end = 0
        new_text = ""
        entities: Sequence[AnonymizationEntity] = []
        for entity in res["Entities"]:
            new_text += text[last_end : entity["BeginOffset"]] + "*" * (
                entity["EndOffset"] - entity["BeginOffset"]
            )
            last_end = entity["EndOffset"]
            classification = CategoryType.choose_category_subcategory(entity["Type"])
            entities.append(
                AnonymizationEntity(
                    offset=entity["BeginOffset"],
                    length=entity["EndOffset"] - entity["BeginOffset"],
                    original_label=entity["Type"],
                    content=text[entity["BeginOffset"] : entity["EndOffset"]],
                    confidence_score=entity["Score"],
                    category=classification["category"],
                    subcategory=classification["subcategory"],
                )
            )
        new_text += text[last_end:]
        standardized_response = AnonymizationDataClass(
            result=new_text, entities=entities
        )
        return ResponseType(
            original_response=res, standardized_response=standardized_response
        )

    def text__entity_sentiment(
        self, text: str, language: str, **kwargs
    ) -> ResponseType:
        payload = {"Text": text, "LanguageCode": language}
        original_response = handle_amazon_call(
            self.clients["text"].detect_targeted_sentiment, **payload
        )

        entity_items: List[Entity] = []
        for entity in original_response["Entities"]:
            for mention in entity["Mentions"]:
                std_entity = Entity(
                    text=mention["Text"],
                    type=Entities.get_entity(mention["Type"]),
                    sentiment=mention["MentionSentiment"]["Sentiment"]
                    .lower()
                    .capitalize(),
                    begin_offset=mention["BeginOffset"],
                    end_offset=mention["EndOffset"],
                )
                entity_items.append(std_entity)
        return ResponseType(
            original_response=original_response,
            standardized_response=EntitySentimentDataClass(items=entity_items),
        )

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
    ) -> ResponseType[GenerationDataClass]:
        # Headers for the HTTP request
        accept_header = "application/json"
        content_type_header = "application/json"

        # Body of the HTTP request, containing text, maxTokens, and temperature
        request_body = json.dumps(
            {
                "inputText": text,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "stopSequences": [],
                    "temperature": temperature,
                },
            }
        )

        # Parameters for the HTTP request
        request_params = {
            "body": request_body,
            "modelId": f"amazon.{model}",
            "accept": accept_header,
            "contentType": content_type_header,
        }
        response = handle_amazon_call(
            self.clients["bedrock"].invoke_model, **request_params
        )
        response_body = json.loads(response.get("body").read())
        generated_text = response_body["results"][0]["outputText"]

        # Calculate number of tokens :
        prompt_tokens = response_body["inputTextTokenCount"]
        completions_tokens = response_body["results"][0]["tokenCount"]
        response_body["usage"] = {"total_tokens": prompt_tokens + completions_tokens}
        standardized_response = GenerationDataClass(generated_text=generated_text)

        return ResponseType[GenerationDataClass](
            original_response=response_body,
            standardized_response=standardized_response,
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str] = None,
        previous_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stream: bool = False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
            text=text,
            previous_history=previous_history,
            chatbot_global_action=chatbot_global_action,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
            **kwargs,
        )
        return response
