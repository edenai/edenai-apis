def code_generation_arguments(provider_name: str):
    return {
        "instruction": "Write a function that checks if a year is a leap year. in python",
        "temperature": 0.1,
        "max_tokens": 1024,
        "prompt": "",
        "settings": {
            "openai": "gpt-4o",
            "google": "gemini-1.5-pro",
            "xai": "grok-2-latest",
        },
    }
