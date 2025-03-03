def chat_arguments(provider_name: str):
    return {
        "model": "mistral/mistral-saba-latest",
        "messages": [
            {"role": "system", "content": "Always reply like a pirate"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image please!"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    },
                ],
            },
        ],
        "temperature": 1,
        "max_tokens": 1000,
    }
