# pylint: disable=locally-disabled, line-too-long
def custom_named_entity_recognition_arguments(provider_name: str):
    return {
        "text": "Yesterday, I met John Smith at Starbucks in New York City. He works for IBM.",
        "entities": ["person", "location", "organization"],
        "examples": None,
        "settings": {"openai": "gpt-4o", "cohere": "command", "xai": "grok-2-latest"},
    }
