import json
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import requests
from edenai_apis.features import TextInterface
from edenai_apis.features.text import PromptOptimizationDataClass
from edenai_apis.features.text.anonymization import AnonymizationDataClass
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationEntity,
)
from edenai_apis.features.text.anonymization.category import CategoryType
from edenai_apis.features.text.moderation.category import CategoryType as CategoryTypeModeration
from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat
from edenai_apis.features.text.code_generation.code_generation_dataclass import (
    CodeGenerationDataClass,
)
from edenai_apis.features.text.custom_classification import (
    CustomClassificationDataClass,
)
from edenai_apis.features.text.custom_named_entity_recognition import (
    CustomNamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.embeddings import EmbeddingDataClass, EmbeddingsDataClass
from edenai_apis.features.text.generation import GenerationDataClass
from edenai_apis.features.text.keyword_extraction import KeywordExtractionDataClass
from edenai_apis.features.text.keyword_extraction.keyword_extraction_dataclass import (
    InfosKeywordExtractionDataClass,
)
from edenai_apis.features.text.moderation import ModerationDataClass, TextModerationItem
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import (
    NamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.prompt_optimization import (
    PromptDataClass,
    PromptOptimizationDataClass,
)
from edenai_apis.features.text.question_answer import QuestionAnswerDataClass
from edenai_apis.features.text.search import InfosSearchDataClass, SearchDataClass
from edenai_apis.features.text.sentiment_analysis import SentimentAnalysisDataClass
from edenai_apis.features.text.spell_check.spell_check_dataclass import (
    SpellCheckDataClass,
    SpellCheckItem,
    SuggestionItem,
)
from edenai_apis.features.text.summarize import SummarizeDataClass
from edenai_apis.features.text.topic_extraction import (
    ExtractedTopic,
    TopicExtractionDataClass,
)
from edenai_apis.utils.conversion import (
    closest_above_value,
    construct_word_list,
    find_all_occurrence,
    standardized_confidence_score,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.metrics import METRICS
from edenai_apis.utils.types import ResponseType
from pydantic_core._pydantic_core import ValidationError

import openai

from .helpers import (
    construct_anonymization_context,
    construct_classification_instruction,
    construct_custom_ner_instruction,
    construct_keyword_extraction_context,
    construct_ner_instruction,
    construct_prompt_optimization_instruction,
    construct_sentiment_analysis_context,
    construct_spell_check_instruction,
    construct_topic_extraction_context,
    get_openapi_response,
    prompt_optimization_missing_information,
)


class OpenaiTextApi(TextInterface):
    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str
    ) -> ResponseType[SummarizeDataClass]:
        if not model:
            model = self.model

        url = f"{self.url}/engines/{model}/completions"
        payload = {
            "prompt": text + "\n\nTl;dr",
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        standardized_response = SummarizeDataClass(
            result=original_response["choices"][0]["text"]
        )

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__moderation(
        self, text: str, language: str
    ) -> ResponseType[ModerationDataClass]:
        try:
            response = requests.post(
                f"{self.url}/moderations", headers=self.headers, json={"input": text}
            )
        except Exception as exc:
            raise ProviderException(str(exc), code=500)
        original_response = get_openapi_response(response)

        classification: Sequence[TextModerationItem] = []
        if result := original_response.get("results", None):
            for key, value in result[0].get("category_scores", {}).items():
                classificator = CategoryTypeModeration.choose_category_subcategory(key)
                classification.append(
                    TextModerationItem(
                        label=key,
                        category=classificator["category"],
                        subcategory=classificator["subcategory"],
                        likelihood_score=value,
                        likelihood=standardized_confidence_score(value)
                    )
                )
        standardized_response: ModerationDataClass = ModerationDataClass(
            nsfw_likelihood=ModerationDataClass.calculate_nsfw_likelihood(
                classification
            ),
            items=classification,
            nsfw_likelihood_score=ModerationDataClass.calculate_nsfw_likelihood_score(
                classification
            )
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
    ) -> ResponseType[SearchDataClass]:
        if model is None:
            model = "1536__text-embedding-ada-002"

        # Import the function
        function_score = METRICS[similarity_metric]

        # Embed the texts & query
        texts_embed_response = OpenaiTextApi.text__embeddings(
            self, texts=texts, model=model
        ).original_response
        query_embed_response = OpenaiTextApi.text__embeddings(
            self, texts=[query], model=model
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

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
    ) -> ResponseType[QuestionAnswerDataClass]:
        if not model:
            model = self.model

        url = f"{self.url}/completions"

        # With search get the top document with the question & construct the context
        document = self.text__search(texts, question).model_dump()
        context = document["standardized_response"]["items"][0]["document"]
        prompt_questions = [
            "\nQ:" + example[0] + "\nA:" + example[1] for example in examples
        ]
        prompts = [
            examples_context
            + "\n\n"
            + "".join(prompt_questions)
            + "\n\n"
            + texts[context]
            + "\nQ:"
            + question
            + "\nA:"
        ]
        payload = {
            "model": model,
            "prompt": prompts,
            "max_tokens": 100,
            "temperature": temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        answer = original_response["choices"][0]["text"].split("\n")
        answer = answer[0]
        standardized_response = QuestionAnswerDataClass(answers=[answer])

        result = ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__anonymization(
        self, text: str, language: str
    ) -> ResponseType[AnonymizationDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_anonymization_context(text)
        payload = {
            "prompt": prompt,
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        data = (
            original_response["choices"][0]["text"]
            .replace("\n\n", " ")
            .replace("\n", "")
            .strip()
        )
        try:
            data_dict = json.loads(rf"{data}")
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")
        new_text = text
        entities: Sequence[AnonymizationEntity] = []
        for entity in data_dict.get("entities", []):
            classificator = CategoryType.choose_category_subcategory(
                entity.get("label")
            )
            offset = closest_above_value(
                find_all_occurrence(text, entity.get("content", "")),
                entity.get("offset", 0),
            )
            length = len(entity.get("content", ""))
            try:
                entities.append(
                    AnonymizationEntity(
                        offset=offset,
                        length=length,
                        content=entity.get("content"),
                        original_label=entity.get("label"),
                        category=classificator["category"],
                        subcategory=classificator["subcategory"],
                        confidence_score=entity.get("confidence_score"),
                    )
                )
            except ValidationError as exc:
                raise ProviderException(
                    "An error occurred while parsing the response."
                ) from exc
            tmp_new_text = new_text[0:offset] + "*" * length
            tmp_new_text += new_text[offset + length :]
            new_text = tmp_new_text

        standardized_response = AnonymizationDataClass(
            result=new_text, entities=entities
        )
        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__keyword_extraction(
        self, language: str, text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_keyword_extraction_context(text)
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        try:
            keywords = json.loads(original_response["choices"][0]["text"]) or {}
            if isinstance(keywords, list) and len(keywords) > 0:
                keywords = keywords[0]
            items = keywords.get("items", []) or []
            items_standardized = []
            for item in items:
                keyword = item.get("keyword")
                try:
                    importance = float(item.get("importance"))
                except:
                    importance = 0
                if not keyword or not isinstance(keyword, str):
                    continue
                item_standardized = InfosKeywordExtractionDataClass(
                    keyword=keyword, importance=importance
                )
                items_standardized.append(item_standardized)
            standardized_response = KeywordExtractionDataClass(items=items_standardized)
        except (KeyError, json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_sentiment_analysis_context(text)
        payload = {
            "prompt": prompt,
            "model": self.model,
            "temperature": 0,
            "logprobs": 1,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        # Create output response
        # Get score
        score = np.exp(original_response["choices"][0]["logprobs"]["token_logprobs"][0])
        standarize = SentimentAnalysisDataClass(
            general_sentiment=original_response["choices"][0]["text"][1:],
            general_sentiment_rate=float(score),
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response, standardized_response=standarize
        )

    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        url = f"{self.url}/completions"

        prompt = construct_topic_extraction_context(text)
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "logprobs": 1,
            "temperature": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        # Create output response
        # Get score
        score = np.exp(original_response["choices"][0]["logprobs"]["token_logprobs"][0])
        categories: Sequence[ExtractedTopic] = []
        categories.append(
            ExtractedTopic(
                category=original_response["choices"][0]["text"],
                importance=float(score),
            )
        )
        standarized_response = TopicExtractionDataClass(items=categories)

        return ResponseType[TopicExtractionDataClass](
            original_response=original_response,
            standardized_response=standarized_response,
        )

    def text__code_generation(
        self, instruction: str, temperature: float, max_tokens: int, prompt: str = ""
    ) -> ResponseType[CodeGenerationDataClass]:
        url = f"{self.url}/chat/completions"
        model = "gpt-3.5-turbo"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Be helpful for code generation",
            },
            {"role": "user", "content": instruction},
        ]
        if prompt:
            messages.insert(1, {"role": "user", "content": prompt})

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        standardized_response = CodeGenerationDataClass(
            generated_text=original_response["choices"][0]["message"]["content"]
        )
        return ResponseType[CodeGenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.url}/completions"

        payload = {
            "prompt": text,
            "model": model,
            "temperature": temperature,
        }
        if max_tokens != 0:
            payload["max_tokens"] = max_tokens

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        standardized_response = GenerationDataClass(
            generated_text=original_response["choices"][0]["text"]
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_named_entity_recognition(
        self, text: str, entities: List[str], examples: Optional[List[Dict]] = None
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.url}/completions"
        built_entities = ",".join(entities)
        prompt = construct_custom_ner_instruction(text, built_entities, examples)
        payload = {
            "prompt": prompt,
            "model": self.model,
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": 250,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)

        # Handle errors
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)

        try:
            original_response = response.json()
            items = json.loads(original_response["choices"][0]["text"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response.", code=500
            ) from exc

        standardized_response = CustomNamedEntityRecognitionDataClass(
            items=items.get("items")
        )

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_classification(
        self, texts: List[str], labels: List[str], examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        url = f"{self.url}/completions"

        prompt = construct_classification_instruction(texts, labels, examples)

        # Build the request
        payload = {
            "prompt": prompt,
            "model": self.model,
            "top_p": 1,
            "max_tokens": 500,
            "temperature": 0,
            "logprobs": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        # Getting labels
        detected_labels = original_response["choices"][0]["text"]

        try:
            json_detected_labels = json.loads(detected_labels)
        except json.JSONDecodeError:
            raise ProviderException("An error occurred while parsing the response.")

        return ResponseType[CustomClassificationDataClass](
            original_response=original_response,
            standardized_response=CustomClassificationDataClass(
                classifications=json_detected_labels["classifications"]
            ),
        )

    def text__spell_check(
        self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_spell_check_instruction(text, language)
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Act As a spell checker, your role is to analyze the provided text and proficiently correct any spelling errors. Accept input texts and deliver precise and accurate corrections to enhance the overall writing quality.",
            },
        )
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        try:
            data = original_response["choices"][0]["message"]["content"]
            corrected_items = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response.",
                code=response.status_code,
            )

        corrections = construct_word_list(text, corrected_items)

        items: Sequence[SpellCheckItem] = []
        for item in corrections:
            items.append(
                SpellCheckItem(
                    text=item["word"],
                    offset=item["offset"],
                    length=item["length"],
                    type=None,
                    suggestions=[
                        SuggestionItem(suggestion=item["suggestion"], score=1.0)
                    ],
                )
            )

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )

    def text__named_entity_recognition(
        self, language: str, text: str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        url = f"{self.url}/completions"

        prompt = construct_ner_instruction(text)

        payload = {
            "n": 1,
            "model": self.model,
            "max_tokens": 500,
            "temperature": 0.0,
            "prompt": prompt,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        try:
            original_items = json.loads(original_response["choices"][0]["text"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=NamedEntityRecognitionDataClass(
                items=original_items["items"]
            ),
        )

    def text__embeddings(
        self, texts: List[str], model: str
    ) -> ResponseType[EmbeddingsDataClass]:
        url = "https://api.openai.com/v1/embeddings"
        model = model.split("__")
        if len(texts) == 1:
            texts = texts[0]
        payload = {
            "input": texts,
            "model": model[1],
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        items: Sequence[EmbeddingsDataClass] = []
        embeddings = original_response["data"]

        for embedding in embeddings:
            items.append(EmbeddingDataClass(embedding=embedding["embedding"]))

        standardized_response = EmbeddingsDataClass(items=items)

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream = False,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:

        messages = [{"role": "user", "content": text}]

        if previous_history:
            for idx, message in enumerate(previous_history):
                messages.insert(
                    idx,
                    {"role": message.get("role"), "content": message.get("message")},
                )

        if chatbot_global_action:
            messages.insert(0, {"role": "system", "content": chatbot_global_action})

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream
        }
        try:
            response = openai.ChatCompletion.create(
                **payload
            )
        except Exception as exc:
            raise ProviderException(str(exc))

        # Standardize the response
        if stream is False:
            generated_text = response["choices"][0]["message"]["content"]
            message = [
                ChatMessageDataClass(role="user", message=text),
                ChatMessageDataClass(role="assistant", message=generated_text),
            ]

            standardized_response = ChatDataClass(
                generated_text=generated_text, message=message
            )

            return ResponseType[ChatDataClass](
                original_response=response,
                standardized_response=standardized_response,
            )
        else:
            stream = (chunk["choices"][0]["delta"].get("content", "") for chunk in response)
            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=stream)
            )

    def text__prompt_optimization(
        self, text: str, target_provider: Literal["openai", "google", "cohere"]
    ) -> ResponseType[PromptOptimizationDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_prompt_optimization_instruction(text, target_provider)
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Act as a Prompt Optimizer for LLMs, you take a description in input and generate a prompt from it.",
            },
        )
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.2,
            "n": 3,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        missing_information_call = requests.post(
            url,
            json={
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_optimization_missing_information(text),
                    }
                ],
            },
            headers=self.headers,
        )
        missing_information_response = get_openapi_response(missing_information_call)

        # Calculate total tokens consumed
        total_tokens_missing_information = missing_information_response["usage"][
            "total_tokens"
        ]
        original_response["usage"][
            "missing_information_tokens"
        ] = total_tokens_missing_information
        original_response["usage"]["total_tokens"] += total_tokens_missing_information

        # Standardize the response
        prompts: Sequence[PromptDataClass] = []

        for generated_prompt in original_response["choices"]:
            prompts.append(
                PromptDataClass(text=generated_prompt["message"]["content"].strip('"'))
            )

        standardized_response = PromptOptimizationDataClass(
            missing_information=missing_information_response["choices"][0]["message"][
                "content"
            ],
            items=prompts,
        )

        return ResponseType[PromptOptimizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
