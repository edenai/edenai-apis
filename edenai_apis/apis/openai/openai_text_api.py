from typing import Dict, List, Literal, Optional, Sequence, Union, Any
import requests
from edenai_apis.features import TextInterface
from edenai_apis.features.text.anonymization import AnonymizationDataClass
from edenai_apis.features.text.chat.chat_dataclass import StreamChat, ChatDataClass
from edenai_apis.features.text.code_generation.code_generation_dataclass import (
    CodeGenerationDataClass,
)
from edenai_apis.features.text.custom_classification import (
    CustomClassificationDataClass,
)
from edenai_apis.features.text.custom_named_entity_recognition import (
    CustomNamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.embeddings import EmbeddingsDataClass
from edenai_apis.features.text.generation import GenerationDataClass
from edenai_apis.features.text.keyword_extraction import KeywordExtractionDataClass
from edenai_apis.features.text.moderation import ModerationDataClass
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
)
from edenai_apis.features.text.summarize import SummarizeDataClass
from edenai_apis.features.text.topic_extraction import TopicExtractionDataClass
from edenai_apis.utils.metrics import METRICS
from edenai_apis.utils.types import ResponseType
from .helpers import (
    construct_prompt_optimization_instruction,
    get_openapi_response,
    prompt_optimization_missing_information,
)


class OpenaiTextApi(TextInterface):

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str, **kwargs
    ) -> ResponseType[SummarizeDataClass]:
        response = self.llm_client.summarize(text=text, model=model, **kwargs)
        return response

    def text__moderation(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[ModerationDataClass]:
        response = self.llm_client.moderation(text=text, **kwargs)
        return response

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
        if model is None:
            model = "1536__text-embedding-ada-002"

        # Import the function
        function_score = METRICS[similarity_metric]

        # Embed the texts & query
        texts_embed_response = OpenaiTextApi.text__embeddings(
            self, texts=texts, model=model, **kwargs
        ).original_response
        query_embed_response = OpenaiTextApi.text__embeddings(
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

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
        **kwargs,
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
            "model": "gpt-3.5-turbo-instruct",
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
        self,
        text: str,
        language: str,
        model: Optional[str] = None,
        provider_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ResponseType[AnonymizationDataClass]:
        response = self.llm_client.pii(text=text, model=model, **kwargs)
        return response

    def text__keyword_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        response = self.llm_client.keyword_extraction(text=text, model=model, **kwargs)
        return response

    def text__sentiment_analysis(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        response = self.llm_client.sentiment_analysis(text=text, model=model, **kwargs)
        return response

    def text__topic_extraction(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[TopicExtractionDataClass]:
        response = self.llm_client.topic_extraction(text=text, model=model, **kwargs)
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
        response = self.llm_client.code_generation(
            instruction=instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **kwargs,
        )
        return response

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
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
        self,
        text: str,
        entities: List[str],
        examples: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        response = self.llm_client.custom_named_entity_recognition(
            text=text, entities=entities, examples=examples, model=model, **kwargs
        )
        return response

    def text__custom_classification(
        self,
        texts: List[str],
        labels: List[str],
        examples: List[List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[CustomClassificationDataClass]:
        response = self.llm_client.custom_classification(
            texts=texts, labels=labels, examples=examples, model=model, **kwargs
        )
        return response

    def text__spell_check(
        self, text: str, language: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[SpellCheckDataClass]:
        response = self.llm_client.spell_check(text=text, model=model, **kwargs)
        return response

    def text__named_entity_recognition(
        self, language: str, text: str, model: Optional[str] = None, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        response = self.llm_client.named_entity_recognition(
            text=text, model=model, **kwargs
        )
        return response

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None, **kwargs
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model.split("__")[1] if "__" in model else model
        response = self.llm_client.embeddings(texts=texts, model=model, **kwargs)
        return response

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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        # self.check_content_moderation(
        #     text=text,
        #     chatbot_global_action=chatbot_global_action,
        #     previous_history=previous_history,
        # )

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

    def text__prompt_optimization(
        self, text: str, target_provider: str, **kwargs
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
