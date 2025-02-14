def topic_extraction_arguments(provider_name: str):
    return {
        "language": "en",
        "text": "That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows.",
        "settings": {"openai": "gpt-4o", "xai": "grok-2-latest"},
    }
