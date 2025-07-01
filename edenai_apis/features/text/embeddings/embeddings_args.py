# pylint: disable=locally-disabled, line-too-long
def embeddings_arguments(provider_name: str):
    return {
        "texts": ["Hello world"],
        "settings": {
            "cohere": "4096__embed-english-v2.0",
            "openai": "1536__text-embedding-ada-002",
            "google": "768__textembedding-gecko",
            "jina": "jina-embeddings-v2-base-en",
            "mistral": "1024__mistral-embed",
            "iointelligence": "BAAI/bge-multilingual-gemma2",
        },
    }
