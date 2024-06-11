def chat_arguments(provider_name: str):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": {"text": "Describe this image please ! "},
                    },
                    {
                        "type": "media_url",
                        "content": {
                            "media_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "media_type": "image/jpeg",
                        },
                    },
                ],
            }
        ],
        "chatbot_global_action": None,
        "temperature": 0,
        "max_tokens": 200,
        "stop_sequences": None,
        "top_k": None,
        "top_p": None,
        "stream": False,
        "settings": {
            "openai": "gpt-4-turbo",
            "anthropic": "claude-3-sonnet-20240229-v1:0",
            "google": "gemini-1.5-pro",
        },
        "provider_params": {},
    }
