# pylint: disable=locally-disabled, line-too-long
def custom_named_entity_recognition_arguments():
    return {
        "text": "Barack Hussein Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
        "entities" : ['person', 'date','region','job','nationality'],
        "examples" : [
            {
                'text' : 'Steve Jobs was the co-founder of Apple Inc., a multinational technology company based in Cupertino, California. He was born in San Francisco, California in 1955, and studied at Reed College before dropping out to start Apple with Steve Wozniak in 1976',
                'entities' : [
                    {"entity": "Steve Jobs", "category": "Person"},
                    {"entity": "Apple Inc", "category": "Organization"},
                    {"entity": "California", "category": "Location"},
                    {"entity": "1955", "category": "Date"}

                ]
            }
        ]
    }
