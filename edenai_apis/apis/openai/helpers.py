from typing import List, Optional
from edenai_apis.utils.exception import ProviderException

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
    Construct context for search task prompt
    """
    return f"<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}"

def construct_classification_instruction(labels) -> str:
    """
    Construct an instruction for a classification task.
    """
    instruction = f"Please classify these texts into the following categories: {', '.join(labels)}."
    log_probs = labels._get_score
    return f"{instruction.strip()}\n\n"

def construct_summarize_context(text) -> str:
    pass

def construct_anonymization_context(text: str, examples : Optional[List[List[str]]] ) -> str:
    pass

def construct_keyword_extraction_context(text: str, examples : Optional[List[List[str]]]) -> str:
    pass

def construct_translation_context(text: str, language_output : str, language_input: str) -> str:
    pass

def check_openai_errors(response : dict):
    if "error" in response:
        raise ProviderException(response["error"]["message"])
