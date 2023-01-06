from abc import abstractmethod
from typing import List, Optional
from edenai_apis.features.text import (
    KeywordExtractionDataClass,
    NamedEntityRecognitionDataClass,
    QuestionAnswerDataClass,
    SentimentAnalysisDataClass,
    SyntaxAnalysisDataClass,
    AnonymizationDataClass,
    SummarizeDataClass,
    SearchDataClass,
    TopicExtractionDataClass,
    GenerationDataClass,
    CustomNamedEntityRecognitionDataClass,
    CustomClassificationDataClass
)
from edenai_apis.utils.types import ResponseType


class TextInterface:
    @abstractmethod
    def text__anonymization(
        self, text: str, language: str
    ) -> ResponseType[AnonymizationDataClass]:
        """
        Anonymize text by hiding every *sensitive* words
        that could be used to identify a person

        Args:
            text (str): text to anonymize
            language (str): text's language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        Extract Keywords from a given text

        Args:
            text (str): text to analyze
            language (str): text's language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        """
        Automatically identifies named entities in a text
        and classifies them into predefined categories.

        Args:
            text (str): text to analyze
            language (str): text's language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str]
    ) -> ResponseType[QuestionAnswerDataClass]:
        """
        Ask question related to given texts and get an answer

        Args:
            texts (List[str]): texts to analyze
            question (str): your query
            temperature (float): value between 0 and 1, represent the degree
                of risk the model will take, Higher values mean the model will
                take more risks and value 0 (argmax sampling) works better for
                scenarios with a well-defined answer.
            examples_context (str): example text serving as context
            examples (List[str]): List of example question/answer pairs
                related to `examples_context`
            model (str, optional): which openai model to use, default to `None`
        """
        raise NotImplementedError

    @abstractmethod
    def text__search(
        self, texts: List[str], query: str, model: Optional[str] = None
    ) -> ResponseType[SearchDataClass]:
        """
        Do sementic search over a set of texts

        Args:
            texts (List[str]): texts to analyze
            query (str): your query
            model (str, optional): which openai model to use, Default to `None`.
        """
        raise NotImplementedError

    @abstractmethod
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        """
        Analyze sentiment of a text

        Args:
            text (str): text to analyze
            language (str): text's language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str],
    ) -> ResponseType[SummarizeDataClass]:
        """
        Summarize a given text in a given number of sentences

        Args:
            text (str): text to analyze
            output_sentences (int): number of sentence in the returned summary
            language (str): text's language code in ISO format
            model (str, optional): which openai model to use, Default to `None`.
        """
        raise NotImplementedError

    ### Syntax analysis
    @abstractmethod
    def text__syntax_analysis(
        self, language: str, text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        """
        Syntax analysis consists principally in highlighting the structure of a text.

        Args:
            text (str): text to analyze
            language (str): text's language code in ISO format
        """
        raise NotImplementedError

    @abstractmethod
    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        """
        Extract Keywords from a given text

        Args:
            text (str): text to analyze
            language (str): text's language code in ISO format
        """
        raise NotImplementedError
    
    @abstractmethod
    def text__generation(
        self, text : str, 
        temperature : float, 
        max_tokens : int,
        model : Optional[str], 
    ) -> ResponseType[GenerationDataClass]:
        """
        Text generation from a given prompt

        Args:
            text (str): your prompt
        """
        raise NotImplementedError
    
    @abstractmethod
    def text__custom_named_entity_recognition(
        self, 
        text : str, 
        entities: List[str]
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        """Custom named entity recognition

        Args:
            text (str): text input
            entities (List[str]): list of entites to detect in the text

        Returns:
            ResponseType[CustomNamedEntityRecognitionDataClass]
        """
        raise NotImplementedError
    
    @abstractmethod
    def text__custom_classification(
        self,
        inputs: List[str],
        labels: List[str],
        examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        """custom text classification

        Args:
            inputs (List[str]): Represents a list of queries to be classified, 
                    each entry must not be empty. The maximum is 32 inputs.
            examples (List[List[str]]): An array of examples to provide context to the model.
                    each example is a text string and its associated label/class.
                    each unique label requires at least 2 examples associated with it

        Returns:
            ResponseType[CustomClassificationDataClass]:
        """

        raise NotImplementedError