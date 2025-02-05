import json
from typing import Optional, List, Dict, Tuple, Union, Literal

import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
)
from edenai_apis.features.text.custom_classification import (
    ItemCustomClassificationDataClass,
    CustomClassificationDataClass,
)
from edenai_apis.features.text.custom_named_entity_recognition import (
    CustomNamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.embeddings import EmbeddingsDataClass, EmbeddingDataClass
from edenai_apis.features.text.generation import GenerationDataClass
from edenai_apis.features.text.search import SearchDataClass, InfosSearchDataClass
from edenai_apis.features.text.spell_check.spell_check_dataclass import (
    SpellCheckDataClass,
)
from edenai_apis.features.text.summarize import SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.metrics import METRICS
from edenai_apis.utils.types import ResponseType
from edenai_apis.llmengine.llm_engine import LLMEngine


class CohereApi(ProviderInterface, TextInterface):
    provider_name = "cohere"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.base_url = "https://api.cohere.ai/"
        self.headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "Cohere-Version": "2022-12-06",
        }
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )

    @staticmethod
    def _calculate_summarize_length(output_sentences: int):
        if output_sentences < 3:
            return "short"
        elif output_sentences < 6:
            return "medium"
        else:
            return "long"

    def text__generation(
        self,
        text: str,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.base_url}generate"

        payload = {
            "prompt": text,
            "model": model,
            "temperature": temperature,
            "stop_sequences": ["--"],
            "frequency_penalty": 0.3,
            "truncate": "END",
        }

        if max_tokens != 0:
            payload["max_tokens"] = max_tokens

        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code >= 500:
            raise ProviderException("Internal Server Error")

        original_response = response.json()

        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        generated_texts = original_response.get("generations")
        standardized_response = GenerationDataClass(
            generated_text=generated_texts[0]["text"]
        )

        # Calculate billed tokens
        billed_units = original_response["meta"]["billed_units"]
        original_response["usage"] = {
            "total_tokens": billed_units["input_tokens"] + billed_units["output_tokens"]
        }

        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_classification(
        self,
        texts: List[str],
        labels: List[str],
        examples: List[Tuple[str, str]],
        model: Optional[str] = None,
    ) -> ResponseType[CustomClassificationDataClass]:
        # Build the request
        url = f"{self.base_url}classify"
        example_dict = []
        for example in examples:
            example_dict.append({"text": example[0], "label": example[1]})
        payload = {
            "inputs": texts,
            "examples": example_dict,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = response.json()

        # Handle provider errors
        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        # Standardization
        classifications = []
        for classification in original_response.get("classifications"):
            classifications.append(
                ItemCustomClassificationDataClass(
                    input=classification["input"],
                    label=classification["prediction"],
                    confidence=classification["confidence"],
                )
            )

        return ResponseType[CustomClassificationDataClass](
            original_response=original_response,
            standardized_response=CustomClassificationDataClass(
                classifications=classifications
            ),
        )

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: Optional[str] = None,
    ) -> ResponseType[SummarizeDataClass]:
        url = f"{self.base_url}summarize"
        length = "long"

        if output_sentences:
            length = CohereApi._calculate_summarize_length(output_sentences)

        payload = {
            "length": length,
            "format": "paragraph",
            "model": model,
            "extractiveness": "low",
            "temperature": 0.0,
            "text": text,
        }

        response = requests.post(url, json=payload, headers=self.headers)
        try:
            original_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal server error", code=500) from exc

        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        standardized_response = SummarizeDataClass(
            result=original_response.get("summary", {})
        )

        # Calculate billed tokens
        billed_units = original_response["meta"]["billed_units"]
        original_response["usage"] = {
            "total_tokens": billed_units["input_tokens"] + billed_units["output_tokens"]
        }

        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_named_entity_recognition(
        self,
        text: str,
        entities: List[str],
        examples: Optional[List[Dict]] = None,
        model: Optional[str] = None,
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        response = self.llm_client.custom_named_entity_recognition(
            text=text, model=model, entities=entities, examples=examples
        )
        return response

    def text__spell_check(
        self, text: str, language: str, model: Optional[str] = None
    ) -> ResponseType[SpellCheckDataClass]:
        response = self.llm_client.spell_check(text=text, model=model)
        return response

    def text__embeddings(
        self, texts: List[str], model: str
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model.split("__")[1] if "__" in model else model
        response = self.llm_client.embeddings(texts=texts, model=model)
        return response

    def text__search(
        self,
        texts: List[str],
        query: str,
        similarity_metric: Literal[
            "cosine", "hamming", "manhattan", "euclidean"
        ] = "cosine",
        model: Optional[str] = None,
    ) -> ResponseType[SearchDataClass]:
        if model is None:
            model = "768__embed-multilingual-v2.0"
        # Import the function
        function_score = METRICS[similarity_metric]

        # Embed the texts & query
        texts_embed_response = self.text__embeddings(
            texts=texts, model=model
        ).original_response
        query_embed_response = self.text__embeddings(
            texts=[query], model=model
        ).original_response

        # Extracts embeddings from texts & query
        texts_embed = list(texts_embed_response["embeddings"])
        query_embed = query_embed_response["embeddings"][0]

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

        # Calculate total tokens
        usage = {
            "total_tokens": texts_embed_response["meta"]["billed_units"]["input_tokens"]
            + query_embed_response["meta"]["billed_units"]["input_tokens"]
        }
        # Build the original response
        original_response = {
            "texts_embeddings": texts_embed_response,
            "embeddings_query": query_embed_response,
            "usage": usage,
        }
        result = ResponseType[SearchDataClass](
            original_response=original_response,
            standardized_response=SearchDataClass(items=sorted_items),
        )
        return result

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
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
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
        )
        return response
