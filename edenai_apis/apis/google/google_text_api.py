from typing import Sequence

from edenai_apis.apis.google.google_helpers import get_tag_name, score_to_sentiment
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
)
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
from edenai_apis.utils.types import ResponseType

from google.cloud.language import Document as GoogleDocument
from google.protobuf.json_format import MessageToDict


class GoogleTextApi(TextInterface):
    def text__named_entity_recognition(
        self, language: str, text: str
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
        response = self.clients["text"].analyze_entities(
            document=document,
            encoding_type="UTF8",
        )

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
        self, language: str, text: str
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
        response = self.clients["text"].analyze_sentiment(
            document=document,
            encoding_type="UTF8",
        )

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
        self, language: str, text: str
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
        response = self.clients["text"].analyze_syntax(
            document=document,
            encoding_type="UTF8",
        )
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
                )
            )

        standardized_response = SyntaxAnalysisDataClass(items=items)

        result = ResponseType[SyntaxAnalysisDataClass](
            original_response=response,
            standardized_response=standardized_response,
        )
        return result

    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        # Create configuration dictionnary
        document = GoogleDocument(
            content=text, type_=GoogleDocument.Type.PLAIN_TEXT, language=language
        )
        # Get Api response
        response = self.clients["text"].classify_text(
            document=document,
        )
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
