import json
from typing import Dict, List, Literal, Optional, Sequence, Union

import requests
from edenai_apis.apis.google.google_helpers import (
    get_access_token,
    get_tag_name,
    handle_google_call,
    score_to_sentiment,
    gemini_request,
    palm_request,
)
from edenai_apis.features.text import (
    ChatDataClass,
    CodeGenerationDataClass,
    GenerationDataClass,
)
from edenai_apis.features.text.chat.chat_dataclass import StreamChat
from edenai_apis.features.text.embeddings.embeddings_dataclass import (
    EmbeddingsDataClass,
)
from edenai_apis.features.text.entity_sentiment.entities import Entities
from edenai_apis.features.text.entity_sentiment.entity_sentiment_dataclass import (
    Entity,
    EntitySentimentDataClass,
)
from edenai_apis.features.text.moderation.category import CategoryType
from edenai_apis.features.text.moderation.moderation_dataclass import (
    ModerationDataClass,
    TextModerationItem,
)
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.search import InfosSearchDataClass, SearchDataClass
from edenai_apis.features.text.sentiment_analysis.sentiment_analysis_dataclass import (
    SegmentSentimentAnalysisDataClass,
    SentimentAnalysisDataClass,
)
from edenai_apis.features.text.syntax_analysis.syntax_analysis_dataclass import (
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
)
from edenai_apis.features.text.text_interface import TextInterface
from edenai_apis.features.text.topic_extraction.topic_extraction_dataclass import (
    ExtractedTopic,
    TopicExtractionDataClass,
)
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.metrics import METRICS
from edenai_apis.utils.parsing import extract
from edenai_apis.utils.types import ResponseType

from google.cloud import language_v1
from google.cloud.language import Document as GoogleDocument
from google.protobuf.json_format import MessageToDict


class GoogleTextApi(TextInterface):
    def text__named_entity_recognition(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Getting response of API
        payload = {"document": document, "encoding_type": "UTF8"}
        response = handle_google_call(self.clients["text"].analyze_entities, **payload)

        # Create output response
        # Convert response to dict
        response = MessageToDict(response._pb)
        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        # Analyse response
        # Getting name of entity, its category and its score of confidence
        if response.get("entities") and isinstance(response["entities"], list):
            for ent in response["entities"]:
                if ent.get("salience"):
                    items.append(
                        InfosNamedEntityRecognitionDataClass(
                            entity=ent["name"],
                            importance=ent.get("salience"),
                            category=ent["type"],
                            #    url=ent.get("metadata", {}).get("wikipedia_url", None),
                        )
                    )

        standardized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=response, standardized_response=standardized_response
        )

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                Array that contain api response and TextSentimentAnalysis
        Object that contains the sentiments and their rates
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )

        # Getting response of API
        payload = {
            "document": document,
            "encoding_type": "UTF8",
        }
        response = handle_google_call(self.clients["text"].analyze_sentiment, **payload)

        # Convert response to dict
        response = MessageToDict(response._pb)
        # Create output response
        items: Sequence[SegmentSentimentAnalysisDataClass] = []
        for segment in response["sentences"]:
            items.append(
                SegmentSentimentAnalysisDataClass(
                    segment=segment["text"].get("content"),
                    sentiment=score_to_sentiment(segment["sentiment"].get("score", 0)),
                    sentiment_rate=abs(segment["sentiment"].get("score", 0)),
                )
            )
        standarize = SentimentAnalysisDataClass(
            general_sentiment=score_to_sentiment(
                response["documentSentiment"].get("score", 0)
            ),
            general_sentiment_rate=abs(response["documentSentiment"].get("score", 0)),
            items=items,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=response, standardized_response=standarize
        )

    def text__syntax_analysis(
        self, language: str, text: str, **kwargs
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                Array containing api response and TextSyntaxAnalysis Object
        that contains the sentiments and their syntax
        """

        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Getting response of API
        payload = {
            "document": document,
            "encoding_type": "UTF8",
        }
        response = handle_google_call(self.clients["text"].analyze_syntax, **payload)
        # Convert response to dict
        response = MessageToDict(response._pb)

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Analysing response
        # Getting syntax detected of word and its score of confidence
        for token in response["tokens"]:
            part_of_speech_tag = {}
            part_of_speech_filter = {}
            part_of_speech = token["partOfSpeech"]
            part_of_speech_keys = list(part_of_speech.keys())
            part_of_speech_values = list(part_of_speech.values())
            for key, prop in enumerate(part_of_speech_keys):
                tag_ = ""
                if "proper" in part_of_speech_keys[key]:
                    prop = "proper_name"
                if "UNKNOWN" not in part_of_speech_values[key]:
                    if "tag" in prop:
                        tag_ = get_tag_name(part_of_speech_values[key])
                        part_of_speech_tag[prop] = tag_
                    else:
                        part_of_speech_filter[prop] = part_of_speech_values[key]

            items.append(
                InfosSyntaxAnalysisDataClass(
                    word=token["text"]["content"],
                    tag=part_of_speech_tag["tag"],
                    lemma=token["lemma"],
                    others=part_of_speech_filter,
                    importance=None,
                )
            )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=response,
            standardized_response=standardized_response,
        )
        return result

    def text__topic_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[TopicExtractionDataClass]:
        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Get Api response
        payload = {
            "document": document,
        }
        response = handle_google_call(self.clients["text"].classify_text, **payload)

        # Create output response
        # Convert response to dict
        original_response = MessageToDict(response._pb)

        # Standardize the response
        categories: Sequence[ExtractedTopic] = []
        for category in original_response.get("categories", []):
            categories.append(
                ExtractedTopic(
                    category=category.get("name"), importance=category.get("confidence")
                )
            )
        standardized_response = TopicExtractionDataClass(items=categories)

        result = ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

        return result

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
    ) -> ResponseType[GenerationDataClass]:
        if "gemini" in model:
            payload = {
                "contents": {
                    "role": "user",
                    "parts": [
                        {
                            "text": text,
                        },
                    ],
                },
                "generationConfig": {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            }
            original_response = gemini_request(
                payload=payload,
                model=model,
                api_key=self.api_settings.get("genai_api_key"),
            )
            generated_text = extract(
                original_response, ["candidates", 0, "content", "parts", 0, "text"]
            )
            if not generated_text:
                raise ProviderException("Empty response", code=200)
        else:
            payload = {
                "instances": [{"prompt": text}],
                "parameters": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            token = get_access_token(self.location)
            original_response = palm_request(
                payload=payload,
                model=model,
                location=self.location,
                token=token,
                project_id=self.project_id,
            )
            generated_text = original_response["predictions"][0]["content"]

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(generated_text=generated_text),
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream=False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[StreamChat, ChatDataClass]]:

        response = self.clients["llm_client"].chat(
            text=text,
            chatbot_global_action=chatbot_global_action,
            previous_history=previous_history,
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

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None, **kwargs
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model.split("__")[1] if "__" in model else model
        response = self.clients["llm_client"].embeddings(
            texts=texts, provider_model_name=f"vertex_ai/{model}", model=model, **kwargs
        )
        return response

    def text__code_generation(
        self,
        instruction: str,
        temperature: float,
        max_tokens: int,
        prompt: str = "",
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[CodeGenerationDataClass]:
        response = self.clients["llm_client"].code_generation(
            instruction=instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return response

    def text__entity_sentiment(self, text: str, language: str, **kwargs):
        client = language_v1.LanguageServiceClient()
        type_ = language_v1.types.Document.Type.PLAIN_TEXT
        document = {"content": text, "type_": type_, "language": language}
        encoding_type = language_v1.EncodingType.UTF8

        payload = {"request": {"document": document, "encoding_type": encoding_type}}
        response = handle_google_call(client.analyze_entity_sentiment, **payload)

        original_response = MessageToDict(response._pb)

        entity_items: List[Entity] = []
        for entity in original_response.get("entities", []):
            for mention in entity["mentions"]:
                sentiment = mention["sentiment"].get("score")
                if sentiment is None:
                    sentiment_score = "Neutral"
                elif sentiment > 0:
                    sentiment_score = "Positive"
                elif sentiment < 0:
                    sentiment_score = "Negative"
                else:
                    sentiment_score = "Neutral"

                begin_offset = mention["text"].get("beginOffset")
                end_offset = None
                if begin_offset:
                    end_offset = mention["text"]["beginOffset"] + len(
                        mention["text"]["content"]
                    )

                std_entity = Entity(
                    text=mention["text"]["content"],
                    type=Entities.get_entity(entity["type"]),
                    sentiment=sentiment_score,
                    begin_offset=begin_offset,
                    end_offset=end_offset,
                )
                entity_items.append(std_entity)

        return ResponseType(
            original_response=original_response,
            standardized_response=EntitySentimentDataClass(items=entity_items),
        )

    def text__moderation(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[ModerationDataClass]:
        """
        :param language:        String that contains the language code
        :param text:            String that contains the text to analyse
        :return:                Array that contain api response and TextSentimentAnalysis
        Object that contains the sentiments and their rates
        """

        # Create configuration dictionnary
        client = language_v1.LanguageServiceClient()
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )

        # Getting response of API
        payload = {"document": document}
        response = handle_google_call(client.moderate_text, **payload)

        # Convert response to dict
        original_response = MessageToDict(response._pb)

        # Create output response
        items: Sequence[TextModerationItem] = []
        for moderation in original_response.get("moderationCategories", []) or []:
            classificator = CategoryType.choose_category_subcategory(
                moderation.get("name")
            )
            items.append(
                TextModerationItem(
                    label=moderation.get("name"),
                    category=classificator["category"],
                    subcategory=classificator["subcategory"],
                    likelihood_score=moderation.get("confidence", 0),
                    likelihood=standardized_confidence_score(
                        moderation.get("confidence", 0)
                    ),
                )
            )
        standardized_response: ModerationDataClass = ModerationDataClass(
            nsfw_likelihood=ModerationDataClass.calculate_nsfw_likelihood(items),
            items=items,
            nsfw_likelihood_score=ModerationDataClass.calculate_nsfw_likelihood_score(
                items
            ),
        )

        return ResponseType[ModerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__search(
        self,
        texts: List[str],
        query: str,
        similarity_metric: Literal[
            "cosine", "hamming", "manhattan", "euclidean"
        ] = "cosine",
        model: str = None,
        **kwargs,
    ) -> ResponseType[SearchDataClass]:
        if len(texts) > 5:
            raise ProviderException(
                "Google does not support search in more than 5 items."
            )
        if model is None:
            model = "768__textembedding-gecko"
        # Import the function
        function_score = METRICS[similarity_metric]

        # Embed the texts & query
        texts_embed_response = GoogleTextApi.text__embeddings(
            self, texts=texts, model=model, **kwargs
        ).original_response
        query_embed_response = GoogleTextApi.text__embeddings(
            self, texts=[query], model=model, **kwargs
        ).original_response

        # Extract Tokens consumed
        texts_usage = texts_embed_response.get("usage").get("total_tokens")
        query_usage = query_embed_response.get("usage").get("total_tokens")

        # Extracts embeddings from texts & query
        texts_embed = [item["embedding"] for item in texts_embed_response.get("data")]
        query_embed = query_embed_response["data"][0]["embedding"]

        items = []
        # Calculate score for each text index
        for index, text in enumerate(texts_embed):
            score = function_score(query_embed, text)
            items.append(
                InfosSearchDataClass(
                    object="search_result", document=index, score=score
                )
            )

        # Sort items by score in descending order
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)

        # Build the original response
        original_response = {
            "texts_embeddings": texts_embed_response,
            "embeddings_query": query_embed_response,
            "usage": {"total_tokens": texts_usage + query_usage},
        }

        result = ResponseType[SearchDataClass](
            original_response=original_response,
            standardized_response=SearchDataClass(items=sorted_items),
        )
        return result
