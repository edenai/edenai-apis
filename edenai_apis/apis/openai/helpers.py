from typing import List, Optional
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
    texts = ''
    labels = ''
    for example in x:
        texts += "{text}\n".format(text=example[0].replace("\n", " ").strip())
        labels += "{label}\n".format(label=example[1].replace("\n"," ").strip())
    return texts +"\n\n Text categories:\n\n"+ labels

def construct_search_context(query, document) -> str:
    """
        Given a query and a document, construct a string that serves as a search context.
        The returned string will contain the document followed by a separator line and a statement 
        indicating the relevance of the document to the query.
        
    """
    return f"<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}"

def construct_classification_instruction(labels) -> str:
    instruction = f"Please classify these texts into the following categories: {', '.join(labels)}."
    return f"{instruction.strip()}\n\n"

def construct_anonymization_context(text: str) -> str:
    prompt = 'identify, categorize, and redact sensitive information in the text below. First write the redacted Text by replacing the entity with [REDACTED], then extract the entity, finally extract the confidence Score between 0.0-1.0.\n\nDesired format:\n{{"redactedText" : "redactedText","entities": [[entity, category, confidence score]]}}\n\n Text:###{data}###\nOutput:'.format(data=text)
    return prompt

def construct_keyword_extraction_context(text: str) -> str:
    prompt = f"""
    You are a highly intelligent and accurate Keyword Extraction system. 
    You take text as input and your task is to returns the key phrases or talking points and a confidence score between 0.0-1.0 to support that this is a key phrase.
    
    Desired format:
            {{"items":[{{"keyword":"keyword","importance":"score"}}]}}
    
    Text: ###{text}###\nOutput:
    """
    return prompt

def construct_translation_context(text: str, source_language : str, target_language: str) -> str:
    prompt = f"Translate the following text from {get_language_name_from_code(source_language)} to {get_language_name_from_code(target_language)}:. text:\n###{text}###\ntranslation:"
    return prompt

def construct_language_detection_context(text: str) -> str:
    prompt = "Detect the ISO 639-1 language (only the code) of this text.\n\n Text: ###{data}###\ISO 639-1:".format(data=text)
    return prompt

def construct_sentiment_analysis_context(text: str) -> str:
    prompt = f"Label the text below with one of these sentiments 'Positive','Negative','Neutral':\n\n text:###"+text+"###\nlabel:"
    return prompt

def construct_topic_extraction_context(text: str)-> str:
    prompt = "What is the main taxonomy of the text below. text:###{data}###put the result in this line:\n\n".format(data=text)
    return prompt

def check_openai_errors(response : dict):
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
        Found the spelling mistakes in the text below in {language}.
        Please create a list of suggests words to replace him and the confidence score between 0.0-1.0.
        We need also a type of misspelling error
        To calculate the start offset of the word you must count all off characters before the word including spaces and punctuation.
        For example: "Hello, world!" the start offset of "world" is 7.

        Desired format:
            {{"items":[{{"text":"word","offset":start_offset,"type":type,"suggestions":[{{"suggestion":"new word","score":value}}]}}]}}

        Text:###{text}###\nOutput:
    """

def construct_ner_instruction(text: str) -> str:
    """
        This function takes a text as input and returns a string that contains the instruction.
    """
    return f"""
        Please extract the entities from the text below and the confidence score between 0.0-1.0.
        We need also the type of the entity.

        Desired format:
            {{"items":[{{"entity":"entity","category":"categrory","importance":score}}]}}

        Text:###{text}###\nOutput:
    """

def construct_custom_ner_instruction(text: str, built_entities) -> str:
    return f"""
    Text : Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company. Originally marketed as a temperance drink and intended as a patent medicine, it was invented in the late 19th century by John Stith Pemberton in Atlanta, Georgia.
    Entities : person, state, drink, date.
    Extracted_Entities: {{
        "items":[
            {{"entity":"John Stith Pemberton", "category":"person"}},
            {{"entity":"Georgia", "category":"state"}},
            {{"entity":"Coca-Cola", "category":"drink"}},
            {{"entity":"coke", "category":"drink"}},
            {{"entity":"19th century", "category":"date"}}
            ]
        }}
    Text : {text}
    Entities :{built_entities}
    Output:
    """