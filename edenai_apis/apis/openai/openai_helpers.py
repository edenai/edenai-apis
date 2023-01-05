
SCORE_MULTIPLIER = 100

def _construct_context_qa(query: str, document: str) -> str:
    return f"<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}"
    

def _get_score(context, query, log_probs, text_offsets) -> float:
    log_prob = 0
    count = 0
    cutoff = len(context) - len(query)

    for i in range(len(text_offsets) - 1, 0, -1):
        log_prob += log_probs[i]
        count += 1

        if text_offsets[i] <= cutoff and text_offsets[i] != text_offsets[i - 1]:
            break

    return log_prob / float(count) * SCORE_MULTIPLIER

def _construct_context_classification(labels) -> str:
    instruction = f"Please classify a piece of text into the following categories: {', '.join(labels)}."
    return f"{instruction.strip()}\n\n"