# pylint: disable=locally-disabled, line-too-long
def named_entity_recognition_arguments(provider_name: str):
    return {
        "language": "en",
        "text": "Barack Hussein Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
        "settings": {"openai": "gpt-4o", "xai": "grok-2-latest"},
    }
