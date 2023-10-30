def chat_arguments(provider_name: str):
    return {
        "text": "Barack Hussein Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
        "chatbot_global_action": "You are a keyword extractor. Extract Only the word from the text provided.",
        "previous_history": [
            {
                "role": "user",
                "message": "Steve Jobs was a co-founder of Apple Inc., a multinational technology company headquartered in Cupertino, California. He was also the CEO and a major shareholder of Pixar Animation Studios, which was later acquired by The Walt Disney Company. Jobs was widely recognized as a visionary entrepreneur and a pioneer in the personal computer industry. In addition to his business ventures, he was also known for his charismatic personality, his signature black turtleneck, and his famous keynote presentations at Apple's product launches.",
            },
            {
                "role": "assistant",
                "message": "steve jobs, apple inc, pixar, california",
            },
        ],
        "temperature": 0.6,
        "max_tokens": 1024,
        "settings": {},
    }
