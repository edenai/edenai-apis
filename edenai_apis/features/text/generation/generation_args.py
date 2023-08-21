def generation_arguments(provider_name: str):
    return {
        "text": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?",
        "max_tokens": 25,
        "temperature": 0.8,
        "settings": {},
    }
