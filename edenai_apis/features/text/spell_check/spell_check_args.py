def spell_check_arguments(provider_name: str):
    return {
        "language": "en",
        "text": "Hollo, wrld! How r yu?",
        "settings": {
            "openai": "gpt-4o",
            "cohere": "command",
            "xai": "grok-2-vision-1212",
        },
    }
