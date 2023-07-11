from typing import List, Optional, Dict
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code

SCORE_MULTIPLIER = 100.0


def get_score(context, query, log_probs, text_offsets) -> float:
    log_prob = 0
    count = 0
    cutoff = len(context) - len(query)

    for i in range(len(text_offsets) - 1, 0, -1):
        log_prob += log_probs[i]
        count += 1

        if text_offsets[i] <= cutoff and text_offsets[i] != text_offsets[i - 1]:
            break

    return log_prob / float(count) * SCORE_MULTIPLIER


def format_example_fn(x: List[List[str]]) -> str:
    examples_formated = ""
    for example in x:
        examples_formated += "Text: {text}\nText Classification: {label}\n\n".format(
            text=example[0].replace("\n", " ").strip(),
            label=example[1].replace("\n", " ").strip(),
        )
    return examples_formated


def format_texts_fn(x: List[str]) -> str:
    texts_formated = ""
    for text in x:
        texts_formated += "Text: {text}\n-----\n".format(
            text=text.replace("\n", " ").strip()
        )
    return texts_formated


def construct_search_context(query, document) -> str:
    """
    Given a query and a document, construct a string that serves as a search context.
    The returned string will contain the document followed by a separator line and a statement
    indicating the relevance of the document to the query.

    """
    return f"<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}"


def construct_classification_instruction(
    texts: list, labels: list, examples: list
) -> str:
    json_format = '{"classifications": [{"input": <text>, "label": <label>, "confidence": <confidence_score>}]}'
    return f"""You should act as a text classifier.
Classify the text below into one of the following categories: {", ".join(labels)}.
The texts is delimited by triple hashtags and separate by -----.

The output should be a json. The format is as follows:
{json_format}

If you cant classify the text into any of the categories, please use the category "other".

Examples:
{format_example_fn(examples)}

Text:
#####
{format_texts_fn(texts)}
#####

Your output:
"""


def construct_anonymization_context(text: str) -> str:
    output_template = '{{"redactedText" : "...", "entities": [{{content: entity, label: category, confidence_score: confidence score, offset: start_offset}}]}}'
    prompt = f"""Please analyze the following text and identify any personal and sensitive information contained within it. For each instance of personal information, please provide a category and a confidence score. Categories could include, but should not be limited to, names, addresses, phone numbers, email addresses, social security numbers, enterprise name, any private information and credit card numbers. Use a confidence score between 0 and 1 to indicate how certain you are that the identified information is actually personal information.
    The text is included between three backticks.

    First write the redacted Text by replacing each character of identified entity with `*`, then extract the entity, finally extract the confidence Score between 0.0-1.0.

    The category must be one of the following: "name", "address", "phonenumber", "email", "social security number", "organization", "credit card number", "other".
    
    Your output should be a json that looks like this : {output_template}

    The text:
    ```{text}```

    Your output:'
    """
    return prompt


def construct_keyword_extraction_context(text: str) -> str:
    output_template = '{{"items":[{{"keyword": ... , "importance":...}}]}}'
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
    prompt = f"Translate the following text from {get_language_name_from_code(source_language)} to {get_language_name_from_code(target_language)}:. text:\n###{text}###\ntranslation:"
    return prompt


def construct_language_detection_context(text: str) -> str:
    prompt = "Detect the ISO 639-1 language (only the code) of this text.\n\n Text: ###{data}###\ISO 639-1:".format(
        data=text
    )
    return prompt


def construct_sentiment_analysis_context(text: str) -> str:
    prompt = (
        f"Label the text below with one of these sentiments 'Positive','Negative','Neutral'.\n\nThe text is delimited by triple backticks text:\n ```"
        + text
        + "```\n\nYour label:"
    )
    return prompt


def construct_topic_extraction_context(text: str) -> str:
    prompt = "What is the main taxonomy of the text below. text:###{data}###put the result in this line:\n\n".format(
        data=text
    )
    return prompt


def check_openai_errors(response: dict):
    """
    This function takes a response from OpenAI API as input and raises a ProviderException if the response contains an error.
    """
    if "error" in response:
        raise ProviderException(response["error"]["message"])


def construct_spell_check_instruction(text: str, language: str) -> str:
    """
    This function takes a text as input and returns a string that contains the instruction.
    """

    return f"""
You should act as a spell and grammar checker.
Find the spelling and grammar mistakes in the text written in {language} delimited by triple hashtags.
Please create a list of suggestions to correct each mistake and the confidence score between 0.0 and 1.0.
You should also return the type of each mistake.
To calculate the start offset of the word you must count all off characters before the word including spaces and punctuation.
For example: "Hello, world!" the start offset of "world" is 7.

The output should be a json that looks like this:
{{"items":[{{"text":"word","offset":start_offset,"type":type,"suggestions":[{{"suggestion":"new word","score":value}}]}}]}}

If no mistake was found, simply return an empty list of items like follows:
{{"items":[]}}

Text:
###{text}###

Output:
    """


def construct_ner_instruction(text: str) -> str:
    """
    This function takes a text as input and returns a string that contains the instruction.
    """
    return f"""Please extract named entities, their types and a confidence score from the text below. The confidence score between 0 and 1.

The text is delimited by triple backticks text.

The output should be a json formatted like follows : {{"items":[{{"entity":"entity","category":"categrory","importance":score}}]}}\n

The input text:
```{text}```

Your output:"
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
    result = f"""Text : {text}
    Entities : {', '.join(set([entity['category'] for entity in extracted_entities]))}
    Extracted Entities: 
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
    instructions = """You need to act like a named entity recognition mode. The user will specify types entities that you need to extract from his text.
    The text is included between three backticks."""

    return (
        instructions
        + prompt_examples
        + f"""
    Text : \n```{text}```\n\n
    Entities : \n{built_entities}\n\n
    Your output:
    """
    )


cohere_prompt_guideines = (
    lambda description: f"""
Construct a prompt for cohere Large language Model from a user description by following these guidelines :

To write a good prompt in the Cohere Playground, you can follow these guidelines based on the provided article:

    Understand the Purpose: Clearly define the purpose of your prompt. Identify what specific output or task you want the generative model to perform.

    Be Clear and Specific: Craft your prompt to provide clear instructions or commands to the model. Use imperative verbs like "generate," "write," "list," or "provide" to guide the model's output. The more specific you are, the better the model will understand your requirements.

    Use Context and Examples: Consider adding context or examples to your prompt to help ground the model's output. Providing additional information or relevant examples can improve the accuracy and relevance of the generated text.

    Iterate and Experiment: Prompt design is a creative process, so feel free to iterate and experiment with different variations of prompts. Test different instructions, formats, or additional details to refine and improve the output.

    Consider Prompt Length: Depending on the complexity of the task, you can use both short and long prompts. A concise prompt can sometimes yield satisfactory results, while longer prompts with more instructions and context may be necessary for more specific tasks.


Remember, prompt design is a combination of science and art. While there are guiding principles, it's also essential to be creative and open to exploring different approaches until you achieve the desired outcome.  
User Description : {description}
Prompt : 
"""
)
google_prompt_guidelines = (
    lambda description: f"""
Construct a prompt for Google Generative Ai Large Language Model from a user description by following these guidelines : 

    Give clear instructions:
    Prompt the model to provide specific guidance and suggestions to enhance the formatting of academic papers. For example:
    "Please provide recommendations to improve the structure and organization of the introduction section of an academic paper on [topic]."

    Include examples:
    Present the model with well-formatted examples of academic paper sections or elements. Request the model to generate similar structures or formats. For instance:
    "Based on the provided abstract, generate a concise and well-structured conclusion for an academic paper."

    Contextual information:
    If needed, provide contextual information that the model can use to tailor its response. For example:
    "Given the research question, provide a discussion section that presents the findings and their implications in a clear and coherent manner."

    Partial input completion:
    Ask the model to complete or revise partial content based on formatting rules. For instance:
    "Given the incomplete citation, please generate the full APA citation for the provided scholarly article."

    Response formatting:
    Guide the model to format its responses appropriately. For example:
    "Format your response as a bulleted list, outlining the key steps involved in conducting a literature review for an academic research paper."

User Description : {description}
Prompt : 
"""
)
openai_prompt_guidelines = (
    lambda description: f"""
Construct a prompt for OpenAI GPT Large Language Model from a user description by following these guidelines : 
To write good prompts for GPT models, you can follow these guidelines :

Be clear and specific: Provide clear instructions and constraints in your prompt to guide the model towards generating the desired output. Avoid leaving ambiguous or open-ended prompts that may result in unexpected responses. Specify the task or query explicitly.

Example: Instead of "Classify this post," use "Classify the sentiment of this post as positive, neutral, or negative: 'My cat is adorable ❤️❤️'"

    Provide sample outputs: If you have specific formatting requirements or want the model to generate outputs in a particular structure, provide examples of the expected output. This helps the model understand the desired format and align its responses accordingly.

Example: Instead of just asking to extract cities and airport codes, provide an example of the expected JSON output structure:

Extract the cities and airport codes from this text as JSON:

Text: "I want to fly from Los Angeles to Miami."
JSON Output: {{
  "Origin": {{
    "CityName": "Los Angeles",
    "AirportCode": "LAX"
  }},
  "Destination": {{
    "CityName": "Miami",
    "AirportCode": "MIA"
  }}
}}

Provide relevant context: If the prompt requires the model to answer questions or perform specific tasks, provide relevant background information or facts to guide the model's understanding. This helps prevent the model from generating fabricated or incorrect responses.

Example: When asking questions about a specific document, include relevant context from that document in your prompt:

Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun... [additional context]

Answer the following question:
Q: Which is the fifth planet from the sun?
A: Jupiter

Q: What’s the mass of Jupiter compared to the Sun?

    Refine, refine, refine: Prompt engineering can be an iterative process. Experiment with different techniques, iterate on your prompts, and learn from the initial outputs generated by the model. Use the generated outputs to provide additional context or guidance in subsequent prompts to improve the quality of the responses.

By following these tips and refining your prompts, you can guide GPT models to generate more accurate and desired completions.

User Description : {description}
Prompt :
"""
)
def construct_prompt_optimization_instruction(
    text: str,
    target_provider: str
    ):
    prompt = {
        "google": google_prompt_guidelines(text),
        "cohere": cohere_prompt_guideines(text),
        "openai": openai_prompt_guidelines(text),
    }

    return prompt[target_provider]
