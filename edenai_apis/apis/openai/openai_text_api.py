import itertools
import json
from typing import Dict, List, Literal, Optional, Sequence, Union
from edenai_apis.features.text.chat.helpers import get_tool_call_from_history_by_id

import openai
import requests
from pydantic_core._pydantic_core import ValidationError

from edenai_apis.features import TextInterface
from edenai_apis.features.text.anonymization import AnonymizationDataClass
from edenai_apis.features.text.anonymization.anonymization_dataclass import (
    AnonymizationEntity,
)
from edenai_apis.features.text.anonymization.category import CategoryType
from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ToolCall,
)
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
from edenai_apis.features.text.moderation.category import (
    CategoryType as CategoryTypeModeration,
)
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
    convert_tool_results_to_openai_tool_calls,
    convert_tools_to_openai,
    finish_unterminated_json,
    get_openapi_response,
    prompt_optimization_missing_information,
)


class OpenaiTextApi(TextInterface):
    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str
    ) -> ResponseType[SummarizeDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = f"""Given the following text, please provide a concise summary in the same language:
        text : {text}
        sumamry : 
        """
        messages = [{"role": "user", "content": prompt}]
        # Build the request
        payload = {
            "model": model,
            "messages": messages,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        standardized_response = SummarizeDataClass(
            result=original_response["choices"][0]["message"]["content"]
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
                        likelihood=standardized_confidence_score(value),
                    )
                )
        standardized_response: ModerationDataClass = ModerationDataClass(
            nsfw_likelihood=ModerationDataClass.calculate_nsfw_likelihood(
                classification
            ),
            items=classification,
            nsfw_likelihood_score=ModerationDataClass.calculate_nsfw_likelihood_score(
                classification
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
            "model": self.model,
            "prompt": prompts,
            "max_tokens": 100,
            "temperature": temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        answers = []
        for choice in original_response["choices"]:
            answer = choice["text"]
            answers.append(answer)
        standardized_response = QuestionAnswerDataClass(answers=answers)

        result = ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__anonymization(
        self, text: str, language: str
    ) -> ResponseType[AnonymizationDataClass]:
        prompt = construct_anonymization_context(text)
        json_output = '{{"redactedText" : "...", "entities": [{{content: entity, label: category, confidence_score: confidence score, offset: start_offset}}]}}'
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as a PII system that takes a text input containing personally identifiable information (PII) and generates an anonymized version of the text, 
                you return a JSON object in this format : {json_output}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
        }
        url = f"{self.url}/chat/completions"
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        pii_data = original_response["choices"][0]["message"]["content"]
        try:
            data_dict = json.loads(rf"{pii_data}")
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
        prompt = construct_keyword_extraction_context(text)
        payload = {
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": self.max_tokens,
            "model": "gpt-3.5-turbo-1106",
            "response_format": {"type": "json_object"},
        }
        try:
            response = openai.ChatCompletion.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        raw_keywords = response["choices"][0]["message"]["content"]
        try:
            if response["choices"][0]["finish_reason"] == "length":
                keywords = json.loads(
                    finish_unterminated_json(raw_keywords, end_brackets="]}")
                )
            else:
                keywords = json.loads(raw_keywords)
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
            original_response=response,
            standardized_response=standardized_response,
        )

    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_sentiment_analysis_context(text)
        json_output = {"general_sentiment": "Positive", "general_sentiment_rate": 0.8}
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as sentiment analysis model capable of analyzing text data and providing a generalized sentiment label.
                you return a JSON object in this format : {json_output}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        sentiments_content = original_response["choices"][0]["message"]["content"]
        try:
            sentiments = json.loads(sentiments_content)
            standarize = SentimentAnalysisDataClass(
                general_sentiment=sentiments["general_sentiment"],
                general_sentiment_rate=sentiments["general_sentiment_rate"],
            )
        except (KeyError, json.JSONDecodeError, ValidationError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response, standardized_response=standarize
        )

    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        url = f"{self.url}/chat/completions"
        prompt = construct_topic_extraction_context(text)
        json_output = {"items": [{"category": "categrory", "importance": 0.9}]}
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as a taxonomy extractor model to automatically identify and categorize hierarchical relationships within a given body of text.
                you return a JSON object in this format : {json_output}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        topics_data = original_response["choices"][0]["message"]["content"]
        try:
            categories_data = json.loads(topics_data)
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        categories = categories_data.get("items", [])

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

        try:
            response = requests.post(url, json=payload, headers=self.headers)
        except requests.exceptions.ChunkedEncodingError:
            raise ProviderException("Connection closed with provider", 400)
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
        built_entities = ",".join(entities)
        prompt = construct_custom_ner_instruction(text, built_entities, examples)
        payload = {
            "messages": [{"role": "system", "content": prompt}],
            "model": "gpt-3.5-turbo-1106",
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        try:
            response = openai.ChatCompletion.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        raw_items = response["choices"][0]["message"]["content"]
        try:
            if response["choices"][0]["finish_reason"] == "length":
                items = json.loads(
                    finish_unterminated_json(raw_items, end_brackets="]}")
                )
            else:
                items = json.loads(raw_items)
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "OpenAI didn't return a valid JSON", code=500
            ) from exc
        for item in items.get("items", []) or []:
            try:
                item["entity"] = str(item.get("entity"))
            except:
                pass

        standardized_response = CustomNamedEntityRecognitionDataClass(
            items=items.get("items")
        )

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=response,
            standardized_response=standardized_response,
        )

    def text__custom_classification(
        self, texts: List[str], labels: List[str], examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        url = f"{self.url}/chat/completions"

        prompt = construct_classification_instruction(texts, labels, examples)
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": """Act as a classification Model, 
                you return a JSON object in this format : {"classifications": [{"input": <text>, "label": <label>, "confidence": <confidence_score>}]}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        # Getting labels
        detected_labels = original_response["choices"][0]["message"]["content"]

        try:
            json_detected_labels = json.loads(detected_labels)
        except json.JSONDecodeError:
            raise ProviderException(
                "An error occurred while parsing the response.", 400
            )

        json_detected_labels_copy = []
        for detected_label in json_detected_labels["classifications"]:
            if detected_label.get("label") and detected_label.get("input"):
                if detected_label.get("confidence") is None:
                    detected_label["confidence"] = 0.0
                json_detected_labels_copy.append(detected_label)

        return ResponseType[CustomClassificationDataClass](
            original_response=original_response,
            standardized_response=CustomClassificationDataClass(
                classifications=json_detected_labels_copy
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
                "content": "Act As a spell checker, your role is to analyze the provided text and proficiently correct any spelling errors. Accept input texts and deliver precise and accurate corrections to enhance the overall writing quality. json",
            },
        )
        payload = {
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
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
            ) from exc

        corrections = construct_word_list(text, corrected_items)

        items: List[SpellCheckItem] = []
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
        url = f"{self.url}/chat/completions"
        prompt = construct_ner_instruction(text)
        json_output = {
            "items": [
                {"entity": "entity", "category": "categrory", "importance": "score"}
            ]
        }
        messages = [{"role": "user", "content": prompt}]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"""Act as a Named Entity Recognition model that indentify and classify named entities within a given text.
                you return a JSON object in this format : {json_output}""",
            },
        )
        # Build the request
        payload = {
            "response_format": {"type": "json_object"},
            "model": "gpt-3.5-turbo-1106",
            "messages": messages,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)
        entities_data = original_response["choices"][0]["message"]["content"]
        try:
            original_items = json.loads(entities_data)
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
        stream=False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        previous_history = previous_history or []
        messages = []
        for msg in previous_history:
            message = {
                "role": msg.get("role"),
                "content": msg.get("message"),
                }
            if msg.get("tool_calls"):
                message['tool_calls'] = [{
                    "id": tool["id"],
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "arguments": tool["arguments"],
                    },
                } for tool in msg['tool_calls']]
            messages.append(message)

        if text and not tool_results:
            messages.append({"role": "user", "content": text})

        if tool_results:
            for tool in tool_results or []:
                tool_call = get_tool_call_from_history_by_id(tool['id'], previous_history)
                try:
                    result = json.dumps(tool["result"])
                except json.JSONDecodeError:
                    result = str(result)
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call['id'],
                    }
                )

        if chatbot_global_action:
            messages.insert(0, {"role": "system", "content": chatbot_global_action})

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if available_tools and not tool_results:
            payload["tools"] = convert_tools_to_openai(available_tools)
            payload["tool_choice"] = tool_choice

        try:
            response = openai.ChatCompletion.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc))

        # Standardize the response
        if stream is False:
            message = response["choices"][0]["message"]
            generated_text = message["content"]
            original_tool_calls = message.get("tool_calls") or []
            tool_calls = []
            for call in original_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
            message = [
                ChatMessageDataClass(
                    role="user", message=text, tools=available_tools
                ),
                ChatMessageDataClass(
                    role="assistant",
                    message=generated_text,
                    tool_calls=tool_calls,
                ),
            ]

            standardized_response = ChatDataClass(
                generated_text=generated_text, message=message
            )

            return ResponseType[ChatDataClass](
                original_response=response,
                standardized_response=standardized_response,
            )
        else:
            stream = (
                ChatStreamResponse(
                    text=chunk["choices"][0]["delta"].get("content", ""),
                    blocked=not chunk["choices"][0].get("finish_reason")
                    in (None, "stop"),
                    provider="openai",
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response=None, standardized_response=StreamChat(stream=stream)
            )

    def text__prompt_optimization(
        self, text: str, target_provider: str
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
