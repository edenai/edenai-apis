import json
from typing import Optional, List, Dict, Sequence, Tuple, Union, Literal, Generator

import requests

from edenai_apis.apis.cohere.helpers import (
    convert_tools_results_to_cohere,
    convert_tools_to_cohere,
    extract_json_text,
)
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ToolCall,
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
    SpellCheckItem,
    SuggestionItem,
)
from edenai_apis.features.text.summarize import SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import construct_word_list
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.metrics import METRICS
from edenai_apis.utils.types import ResponseType


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

    @staticmethod
    def _calculate_summarize_length(output_sentences: int):
        if output_sentences < 3:
            return "short"
        elif output_sentences < 6:
            return "medium"
        else:
            return "long"

    @staticmethod
    def _format_custom_ner_examples(example: Dict):
        # Get the text
        text = example["text"]

        # Get the entities
        entities = example["entities"]

        # Create an empty list to store the extracted entities
        extracted_entities = []

        # Loop through the entities and extract the relevant information
        for entity in entities:
            category = entity["category"]
            entity_name = entity["entity"]

            # Append the extracted entity to the list
            extracted_entities.append({"entity": entity_name, "category": category})

        # Create the string with the extracted entities
        return f"""
        Categories: {', '.join(set([entity['category'] for entity in extracted_entities]))}

        Text: {text}

        Answer: ```json[{', '.join([f'{{"entity":"{entity["entity"]}", "category":"{entity["category"]}"}}' for entity in extracted_entities])}]```

        """

    @staticmethod
    def _format_spell_check_prompt(text: str) -> str:
        return f"""
Given a text with spelling errors, identify the misspelled words and correct them.
Return the results as a json list of objects, where each object contains two keys: "word" and "correction".
The "word" key should contain the misspelled word, and the "correction" key should contain the corrected version of the word.
Return the json response between ```json and ```.

For example, if the misspelled word is 'halo', the corresponding dictionary should be: {{"word": "halo", "correction": "hello"}}.
Text: {text}
Examples of entry Text with misspelling: "Hallo my friend hw are you"
Examples of response: ```json[{{"word": "Hallo", "correction": "hello"}}, {{"word": "hw", "correction": "how"}}]```
List of corrected words:
"""

    @staticmethod
    def __text_to_json(
        lst_data: List[str],
    ) -> Generator[ChatStreamResponse, None, None]:
        lst_json = []
        for token in lst_data:
            if token != "":
                lst_json.append(json.loads(token))
        for elt in lst_json:
            if elt["event_type"] == "text-generation":
                yield ChatStreamResponse(
                    text=elt["text"], blocked=False, provider="cohere"
                )

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
        self, texts: List[str], labels: List[str], examples: List[Tuple[str, str]]
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
        self, text: str, entities: List[str], examples: Optional[List[Dict]] = None
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.base_url}chat"

        # Construct the prompt
        built_entities = ",".join(entities)
        prompt_examples = ""
        if examples is not None:
            for example in examples:
                prompt_examples = (
                    prompt_examples + CohereApi._format_custom_ner_examples(example)
                )
        else:
            prompt_examples = self._format_custom_ner_examples(
                {
                    "text": "Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company. Originally marketed as a temperance drink and intended as a patent medicine, it was invented in the late 19th century by John Stith Pemberton in Atlanta, Georgia. Extracted these entities from the Text if they exist: drink, date",
                    "entities": [
                        {"entity": "Coca-Cola", "category": "drink"},
                        {"entity": "coke", "category": "drink"},
                        {"entity": "19th century", "category": "date"},
                    ],
                }
            )
        prompt = f"""You act as a named entities recognition model.
Extract an exhaustive list of Entities from the given Text according to the specified Categories and return the list as a valid JSON.

return the json response between ```json and ```. The keys of each objects in the list are `entity` and `category`.
`entity` value must be the extracted entity from the text, `category` value must be the category of the extracted entity.
The JSON MUST be valid and conform to the given description.
Be correct and concise. If no entities are found, return an empty list.

Categories: {built_entities}

Text: {text}

For Example:
{prompt_examples}

Your answer:
"""

        # Construct request
        payload = {
            "model": "command",
            "message": prompt,
            "temperature": 0,
            "stop_sequences": ["--"],
            "truncate": "END",
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)

        original_response = response.json()
        data = original_response.get("text")

        print(data)
        try:
            items = extract_json_text(data)
        except json.JSONDecodeError as exc:
            raise ProviderException("Cohere didn't return valid JSON object") from exc

        standardized_response = CustomNamedEntityRecognitionDataClass(items=items)

        # Calculate billed tokens
        billed_units = original_response["meta"]["billed_units"]
        original_response["usage"] = {
            "total_tokens": billed_units["input_tokens"] + billed_units["output_tokens"]
        }

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__spell_check(
        self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        url = f"{self.base_url}chat"

        payload = {
            "message": CohereApi._format_spell_check_prompt(text),
            "model": "command",
            "temperature": 0,
            "stop_sequences": ["--"],
            "truncate": "END",
        }

        response = requests.post(url, json=payload, headers=self.headers)
        original_response = response.json()

        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        try:
            data = extract_json_text(original_response["text"])
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        corrections = construct_word_list(text, data)
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

        # Calculate billed tokens
        billed_units = original_response["meta"]["billed_units"]
        original_response["usage"] = {
            "total_tokens": billed_units["input_tokens"] + billed_units["output_tokens"]
        }
        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )

    def text__embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> ResponseType[EmbeddingsDataClass]:
        url = f"{self.base_url}embed"
        model = model.split("__")[1]
        payload = {"texts": texts, "model": model}
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code >= 500:
            raise ProviderException("Internal Server Error")

        original_response = response.json()
        if "message" in original_response:
            raise ProviderException(
                original_response["message"], code=response.status_code
            )

        items: Sequence[EmbeddingDataClass] = []
        for prediction in original_response["embeddings"]:
            items.append(EmbeddingDataClass(embedding=prediction))

        # Calculate billed tokens
        billed_units = original_response["meta"]["billed_units"]
        original_response["usage"] = {"total_tokens": billed_units["input_tokens"]}
        standardized_response = EmbeddingsDataClass(items=items)
        return ResponseType[EmbeddingsDataClass](
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
        tool_results: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        messages = [{"role": "USER", "message": text}]
        previous_history = previous_history or []
        messages = [
            {"role": msg.get("role"), "message": msg.get("message") or None}
            for msg in previous_history
        ]

        if chatbot_global_action:
            messages.insert(0, {"role": "CHATBOT", "message": chatbot_global_action})

        payload = {
            "message": text,
            "temperature": temperature,
            "model": model,
            "chat_history": messages,
            "stream": stream,
        }


        if tool_results:
            payload["tool_results"] = convert_tools_results_to_cohere(tool_results, previous_history)
            del payload['chat_history']
            payload['message'] = next(filter(lambda msg: msg['role'] == 'USER', previous_history))['message']

        if available_tools:
            payload["tools"] = convert_tools_to_cohere(available_tools)
            if tool_choice == "required":
                payload["preamble"] = "You must choose at least one tool among the available tools"
            elif tool_choice == "none":
                payload["preamble"] = (
                    "You must directly answer the question, please ignore the available tools"
                )
            else:
                payload["preamble"] = (
                    "When a question is irrelevant or unrelated to the available tools, please choose to directly answer it."
                )

        if not available_tools and not tool_results:
            payload["connectors"] = [{"id": "web-search"}]

        response = requests.post(f"{self.base_url}chat", headers=self.headers, json=payload)

        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)

        else:
            if not stream:
                try:
                    original_response = response.json()
                except requests.JSONDecodeError as exp:
                    raise ProviderException(
                        response.text, code=response.status_code
                    ) from exp

                generated_text = original_response["text"]
                tool_calls = []
                generation_id = original_response['generation_id']
                for index, tool in enumerate(original_response.get("tool_calls", [])):
                    tool_id = f"{generation_id}-{tool['name']}-{index}"
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            name=tool["name"],
                            arguments=json.dumps(tool["parameters"]),
                        )
                    )

                message = [
                    ChatMessageDataClass(
                        role="USER", message=text, tools=available_tools
                    ),
                    ChatMessageDataClass(
                        role="CHATBOT", message=generated_text, tool_calls=tool_calls
                    ),
                ]
                standardized_response = ChatDataClass(
                    generated_text=generated_text, message=message
                )

                return ResponseType[ChatDataClass](
                    original_response=original_response,
                    standardized_response=standardized_response,
                )
            else:
                data = response.text
                lst_data = data.split("\n")
                return ResponseType[StreamChat](
                    original_response=None,
                    standardized_response=StreamChat(
                        stream=self.__text_to_json(lst_data)
                    ),
                )
