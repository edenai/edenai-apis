def responses_arguments(provider_name: str):
    return {
        "input": "Tell me a short joke.",
        "model": "gpt-4o-mini",
        "temperature": 1,
        "max_output_tokens": 1000,
    }
