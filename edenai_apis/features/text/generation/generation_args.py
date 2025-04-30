def generation_arguments(provider_name: str):
    return {
        "text": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?",
        "max_tokens": 500,
        "temperature": 0.8,
        "settings": {
            "amazon": "titan-tg1-large",
            "google": "text-bison",
            "openai": "gpt-3.5-turbo-instruct",
            "cohere": "command-nightly",
            "anthropic": "claude-v2",
            "mistral": "large-latest",
            "ai21labs": "j2-ultra",
            "meta": "llama3-1-70b-instruct-v1:0",
            "xai": "grok-2-latest",
            "tenstorrent": "tenstorrent/Meta-Llama-3.3-70B-Instruct",
        },
    }
