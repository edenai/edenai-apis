import json
import aiohttp
from enum import Enum
from typing import List, Optional, Dict

from requests import Response

from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code


class OpenAIErrorCode(Enum):
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


def construct_classification_instruction(
    texts: list, labels: list, examples: list
) -> str:
    formated_examples = formatted_examples_data(examples)
    formated_texts = formatted_text_classification(texts)
    return f"""
        Given the following list of labels and corresponding text, classify each text according to the appropriate label. Here are some examples:

        Labels: {labels}
        Text: {texts}

        Example Classification:
        {formated_examples}

        Now, classify the sentiment of the following texts:

        {formated_texts}
    """


# Function to format and print the examples in custom classification
def formatted_examples_data(data):
    formatted_output = ""
    for i, item in enumerate(data, start=1):
        text = item[0]
        label = item[1]
        formatted_output += f'{i}. Text: "{text}" - Label: "{label}"\n'

    return formatted_output


# Function to format and print the texts in custom classification
def formatted_text_classification(data):
    formated_texts = ""
    for i, item in enumerate(data, start=1):
        formated_texts += f"""{i}. '{item}' """
    return formated_texts


def construct_anonymization_context(text: str) -> str:
    prompt = f"""Please analyze the following text and identify any personal and sensitive information contained within it. For each instance of personal information, please provide a category and a confidence score. Categories could include, but should not be limited to, names, addresses, phone numbers, email addresses, social security numbers, enterprise name, any private information and credit card numbers. Use a confidence score between 0 and 1 to indicate how certain you are that the identified information is actually personal information.
    The text is included between three backticks.

    First write the redacted Text by replacing each character of identified entity with `*`, then extract the entity, finally extract the confidence Score between 0.0-1.0.

    The category must be one of the following: "name", "address", "phonenumber", "email", "social security number", "organization", "credit card number", "other".
    
    The text:
    ```{text}```

    Your output:'
    """
    return prompt


def construct_keyword_extraction_context(text: str) -> str:
    output_template = '{"items":[{"keyword": ... , "importance": ...}]}'
    prompt = f"""
    You are a highly intelligent and accurate Keyword Extraction system.
    You take text as input and your task is to returns the key phrases or talking points and a confidence score between 0.0-1.0 to support that this is a key phrase.
    The text is written between three backticks.
    
    Your output should be a json that looks like: {output_template} 
    
    Text:
    ```{text}```

    Your Output:
    """
    return prompt


def construct_translation_context(
    text: str, source_language: str, target_language: str
) -> str:
    return f"""
    Translate the following text from {get_language_name_from_code(source_language)} to {get_language_name_from_code(target_language)} 
    The text is written between three backticks.
    Text:
    ```{text}```

    Your Output:
"""


def construct_language_detection_context(text: str) -> str:
    return f"""
Analyze the following text and determine the ISO 639-1 language code: 
Please provide only the ISO 639-1 code as the output.
The text is written between three backticks.
```{text}```
"""


def construct_sentiment_analysis_context(text: str) -> str:
    return f"""
    Analyze the following text and label it with one of these sentiments 'Postive', 'Negative', 'Neutral'.
    The text is written between three backticks.
    ```{text}```
"""


def construct_topic_extraction_context(text: str) -> str:
    return f"""
    Analyze the following text and identify its main taxonomy.
    The text is written between three backticks.
    
    ```{text}```
    """


def get_openapi_response(response: Response):
    """
    This function takes a requests.Response as input and return it's response.json()
    raises a ProviderException if the response contains an error.
    """
    try:
        original_response = response.json()
        if "error" in original_response or response.status_code >= 400:
            message_error = original_response["error"]["message"]
            raise ProviderException(message_error, code=response.status_code)
        return original_response
    except Exception:
        raise ProviderException(response.text, code=response.status_code)


# def construct_spell_check_instruction(text: str, language: str) -> str:
#     """
#     This function takes a text as input and returns a string that contains the instruction.
#     """

#     return f"""
# You should act as a spell and grammar checker.
# Find the spelling and grammar mistakes in the text written in {language} delimited by triple hashtags.
# Please create a list of suggestions to correct each mistake and the confidence score between 0.0 and 1.0.
# You should also return the type of each mistake.
# To calculate the start offset of the word you must count all off characters before the word including spaces and punctuation.
# For example: "Hello, world!" the start offset of "world" is 7.

# The output should be a json that looks like this:
# {{"items":[{{"text":"word","offset":start_offset,"type":type,"suggestions":[{{"suggestion":"new word","score":value}}]}}]}}

# If no mistake was found, simply return an empty list of items like follows:
# {{"items":[]}}

# Text:
# ###{text}###


# Output:
#     """
def construct_spell_check_instruction(text: str, language: str):
    return f"""
Given the following text between written in {language} and delimited by triple hashtags, identify and correct any spelling mistakes. Return the results always as a list of dictionaries, each containing the original incorrect word and the corrected word. The dictionaries should be structured as follows: {{ "corrections": [{{"word": "incorrect spelling", "correction": "correct spelling"}}] }}.

Text: ###{text}###
"""


def construct_ner_instruction(text: str) -> str:
    """
    This function takes a text as input and returns a string that contains the instruction.
    """
    example_output = [
        {"Entity": "Barack Obama", "Type": "Person", "Confidence": 0.98},
        {"Entity": "44th President", "Type": "Position", "Confidence": 0.95},
        {"Entity": "United States", "Type": "Country", "Confidence": 0.99},
    ]
    return f"""Please extract named entities, their types, and assign a confidence score between 0 and 1 from the following text. The text is enclosed within triple backticks.

Input Text:
```{text}```

For example, if the input text is: 
```Barack Obama was the 44th President of the United States.```

Your output should be in the following format:
```
    {example_output}
```
    """


def format_custom_ner_examples(example: Dict):
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
    result = f"""
    Text:
    {text}

    Entities:
    {', '.join(set([entity['category'] for entity in extracted_entities]))}


    Output:
    {{
        "items":[
            {', '.join([f'{{"entity":"{entity["entity"]}", "category":"{entity["category"]}"}}' for entity in extracted_entities])}
            ]
    }}
    """

    return result


def construct_custom_ner_instruction(
    text: str, built_entities: str, examples: Optional[List[Dict]]
) -> str:
    if examples:
        prompt_examples = ""
        for example in examples:
            prompt_examples = prompt_examples + format_custom_ner_examples(example)
    else:
        prompt_examples = f"""
        Text : Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company. Originally marketed as a temperance drink and intended as a patent medicine, it was invented in the late 19th century by John Stith Pemberton in Atlanta, Georgia.
        Extracted these entities from the Text if they exist: drink, date
        {{
            "items":[
                {{"entity":"Coca-Cola", "category":"drink"}},
                {{"entity":"coke", "category":"drink"}},
                {{"entity":"19th century", "category":"date"}}
                ]
        }}
    """
    return f"""
You need to act like a named entity recognition model.
Given the following list of entity types and a text, extract all the entities from the text that correspond to the entity types. Ensure that the same entity/category pair is not extracted twice. The output should be in the following JSON format:

Entity Types: {built_entities}
Text: {text}

Expected JSON Output:
{{
  "items": [
    {{"category": "<entity type from the entity list>", "entity": "<extracted entity from the text>"}},
    ...
  ]
}}

For Example:
{examples}


Please perform this task and provide the JSON output.
"""


prompt_optimization_missing_information = (
    lambda user_description: f"""
You are a Prompt Optimizer for LLMs, you take a description in input and generate a prompt from it.

The user's project description is delimited by triple back-ticks.

You should to tell him to provide you what any missing information about his project would guide you towards giving him a better prompt.

If the description is clear don't provide any missing information.

The User's description :

```{user_description}```

missing information : 
"""
)


def finish_unterminated_json(json_string: str, end_brackets: str) -> str:
    """
    take a cut json_string and try to terminate it

    Arguments:
      json_string(str): JSON string to terminate
      end_brackets(str): string representing the ending brackets. eg: '}]'

    Returns:
      str: valid json string

    Raise:
      json.JSONDecodeERror: if couldn't terminate string

    Example:
      >>> finish_unterminated_json('{"data": {"place": "Italy"')
      >>> '{"data": {"place": "Italy"}}
    """
    if not json_string:
        raise json.JSONDecodeError("JSON string couldn't be parsed or finished")

    try:
        new_json_string = json_string + end_brackets
        json.loads(new_json_string)
        return new_json_string
    except json.JSONDecodeError:
        json_string = json_string[:-1]
        return finish_unterminated_json(json_string, end_brackets)


def convert_tools_to_openai(tools: Optional[List[dict]]):
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({"type": "function", "function": tool})
    return openai_tools


def convert_tool_results_to_openai_tool_calls(tools_results: List[dict]):
    result = []
    for tool in tools_results:
        result.append(
            {
                "id": tool["call"]["id"],
                "type": "function",
                "function": {
                    "name": tool["call"]["name"],
                    "arguments": tool["call"]["arguments"],
                },
            }
        )
    return result


async def get_openapi_response_async(response: aiohttp.ClientResponse):
    """
    This function takes an aiohttp.ClientResponse as input and returns its response.json()
    raises a ProviderException if the response contains an error.
    """
    try:
        original_response = await response.json()
        if "error" in original_response or response.status >= 400:
            code = original_response["error"]["code"]
            if code == OpenAIErrorCode.RATE_LIMIT_EXCEEDED.value:
                return None
            message_error = original_response["error"]["message"]
            raise ProviderException(message_error, code=response.status)
        return original_response
    except Exception:
        raise ProviderException(await response.text(), code=response.status)
