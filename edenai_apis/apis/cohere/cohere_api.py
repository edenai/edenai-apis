from typing import Optional, List, Dict, Sequence
import requests
from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    GenerationDataClass,
    ItemCustomClassificationDataClass,
    CustomClassificationDataClass,
    SummarizeDataClass,
    CustomNamedEntityRecognitionDataClass,
)
from edenai_apis.features.text.spell_check.spell_check_dataclass import (
    SpellCheckDataClass,
    SpellCheckItem,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.conversion import closest_above_value, find_all_occurrence
import json


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

    def _calculate_summarize_length(output_sentences: int):
        if output_sentences < 3:
            return "short"
        elif output_sentences < 6:
            return "medium"
        elif output_sentences > 6:
            return "long"

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
        Text: #{text}#
        Answer: "[{', '.join([f'{{"entity":"{entity["entity"]}", "category":"{entity["category"]}"}}' for entity in extracted_entities])}]"
        ---
            """

    def _format_spell_check_prompt(text: str, language: str) -> str:
        return f"""
Find the spelling and grammar mistakes by creating a list of suggestions to correct each mistake and the confidence score between 0.0 and 1.0, it will return the type of each mistake.
To calculate the start offset of the word you must count all off characters before the word including spaces and punctuation, in the text written in samples text.
--
Sample: Hollo wrld! Haw are yu?
Corrected text: {{"items": [{{"text": "Hollo", "type": "typo","offset": 0,"length": 5,"suggestions": [{{"suggestion": "Hello", "score": 1}}]}},{{"text": "wrld","type": "typo","offset": 6,"length": 4, "suggestions": [{{"suggestion": "world", "score": 1 }}] }},{{"text": "Haw", "type": "typo","offset": 12,"length": 3,"suggestions": [{{"suggestion": "How","score": 1}}]}}, {{"text": "yu","type": "typo","offset": 20,"length": 2,"suggestions": [ {{ "suggestion": "you","score": 1}}]}}]}}

--
Sample:  {text}
Corrected text:
"""

    def text__generation(
        self,
        text: str,
        max_tokens: int,
        temperature: float,
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

        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "message" in original_response:
            raise ProviderException(original_response["message"])

        generated_texts = original_response.get("generations")
        standardized_response = GenerationDataClass(
            generated_text=generated_texts[0]["text"]
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_classification(
        self, texts: List[str], labels: List[str], examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        # Build the request
        url = f"{self.base_url}classify"
        example_dict = []
        for example in examples:
            example_dict.append({"text": example[0], "label": example[1]})
        payload = {
            "inputs": texts,
            "examples": example_dict,
            "model": "large",
        }
        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        # Handle provider errors
        if "message" in original_response:
            raise ProviderException(original_response["message"])

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
        self, text: str, output_sentences: int, language: str, model: str
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

        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "message" in original_response:
            raise ProviderException(original_response["message"])

        standardized_response = SummarizeDataClass(
            result=original_response.get("summary", {})
        )

        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__custom_named_entity_recognition(
        self, text: str, entities: List[str], examples: Optional[List[Dict]] = None
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.base_url}generate"

        # Construct the prompt
        built_entities = ",".join(entities)
        prompt_examples = ""
        if examples is not None:
            for example in examples:
                prompt_examples = (
                    prompt_examples + CohereApi._format_custom_ner_examples(example)
                )
        prompt = f"""You act as a named entities recognition model. Extract the specified entities ({built_entities}) from the text enclosed in hash symbols (#) and return a JSON List of dictionaries with two keys: "entity" and "category". The "entity" key represents the detected entity and the "category" key represents the category of the entity.

If no entities are found, return an empty list.

Example :

{prompt_examples}

Text: 
{text}

Answer:"""

        # Construct request
        payload = {
            "model":'command',
            "prompt":prompt,
            "max_tokens":650,
            "temperature":0,
            "k":0,
            "frequency_penalty": 0.3,
            "truncate": "END",
            "stop_sequences":[],
            "return_likelihoods":'NONE'
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)

        original_response = response.json()
        try:
            data = original_response.get("generations")[0]["text"]
            items = json.loads(data)
        except (IndexError, KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        standardized_response = CustomNamedEntityRecognitionDataClass(items=items)

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def text__spell_check(
        self, text: str, language: str
    ) -> ResponseType[SpellCheckDataClass]:
        url = f"{self.base_url}generate"

        payload = {
            "prompt": CohereApi._format_spell_check_prompt(text, language),
            "model": "command-nightly",
            "max_tokens": 1000,
            "temperature": 0,
            "stop_sequences": ["--"],
            "truncate": "END",
        }

        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "message" in original_response:
            raise ProviderException(original_response["message"])

        generated_texts = original_response.get("generations")
        try:
            original_items = json.loads(generated_texts[0]["text"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        items: Sequence[SpellCheckItem] = []
        for item in original_items["items"]:
            try:
                real_offset = closest_above_value(
                    find_all_occurrence(text, item["text"]), item["offset"]
                )
            except ValueError:
                real_offset = item["offset"]
            items.append(
                SpellCheckItem(
                    text=item["text"],
                    offset=real_offset,
                    length=len(item["text"]),
                    type=item["type"],
                    suggestions=item["suggestions"],
                )
            )
        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items),
        )
