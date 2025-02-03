def anonymization_arguments(provider_name: str):
    return {
        "text": "The phone number of Luc is the 06 21 32 43 54.",
        "language": "en",
        "settings": {"openai": "gpt-4o", "xai": "grok-2-vision-1212"},
    }
