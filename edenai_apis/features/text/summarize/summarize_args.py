# pylint: disable=locally-disabled, line-too-long
def summarize_arguments(provider_name: str):
    return {
        "output_sentences": 3,
        "text": "Barack Hussein Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
        "language": "en",
        "settings": {
            "openai": "gpt-4",
            "nlpcloud": "finetuned-llama-2-70b",
            "anthropic": "claude-3-5-sonnet-latest",
            "xai": "grok-2-latest",
        },
    }
