import pytest

from edenai_apis.loaders.data_loader import load_output


@pytest.fixture
def mocked_chat_params(model: str = "gpt-4o-mini"):
    params = {
        "text": "hey how are you ? ",
        "chatbot_global_action": "Act as an assistant",
        "previous_history": [],
        "temperature": 0,
        "max_tokens": 120,
        "model": model,
        "stream": False,
        "available_tools": None,
        "tool_choice": "auto",
        "tool_results": None,
        "mock_response": "Hey, this is the testing machine",
    }
    return params


@pytest.fixture
def mocked_chat_stream_params(model: str = "gpt-4o-mini"):
    params = {
        "text": "hey how are you ? ",
        "chatbot_global_action": "Act as an assistant",
        "previous_history": [],
        "temperature": 0,
        "max_tokens": 120,
        "model": model,
        "stream": True,
        "available_tools": None,
        "tool_choice": "auto",
        "tool_results": None,
        "mock_response": "Hey, this is the testing machine",
    }
    return params


@pytest.fixture
def mocked_multimodal_chat_params(model: str = "gpt-4o"):
    params = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": {"text": "Describe this image please ! "},
                    },
                ],
            }
        ],
        "chatbot_global_action": "act as an assistant",
        "temperature": 0,
        "max_tokens": 25,
        "model": model,
        "stream": False,
        "mock_response": "hey hey",
    }
    return params


@pytest.fixture
def mocked_multimodal_chat_stream_params(model: str = "gpt-4o"):
    params = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": {"text": "Describe this image please ! "},
                    },
                ],
            }
        ],
        "chatbot_global_action": "act as an assistant",
        "temperature": 0,
        "max_tokens": 25,
        "model": model,
        "stream": True,
        "mock_response": "hey hey",
    }
    return params


@pytest.fixture
def mocked_summarize_params(model: str = "gpt-4o"):
    params = {
        "text": "summarize this document..",
        "model": model,
        "mock_response": '{"result": "summarized document"}',
    }
    return params


@pytest.fixture
def mocked_keyword_extraction_params():
    params = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "keyword_extraction")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_topic_extraction_params(model: str = "gpt-4o"):
    params = {
        "text": "example text",
        "model": model,
        "mock_response": load_output("openai", "text", "topic_extraction")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_spell_check_params():
    params = {
        "text": "This is a testt of the splel chek feature.",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "spell_check")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_named_entity_recognition_params():
    params = {
        "text": "Apple was founded by Steve Jobs in California.",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "named_entity_recognition")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_pii_params():
    params = {
        "text": "My name is John Doe and my email is john.doe@example.com.",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "anonymization")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_code_generation_params():
    params = {
        "instruction": "Write a Python function to calculate the factorial of a number.",
        "temperature": 0.7,
        "max_tokens": 150,
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "code_generation")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_custom_classification_params():
    params = {
        "texts": ["This is text one.", "This is text two."],
        "labels": ["Label 1", "Label 2"],
        "examples": [["Example text 1", "Label 1"], ["Example text 2", "Label 2"]],
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "custom_classification")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_custom_ner_params():
    params = {
        "text": "Apple was founded by Steve Jobs in California.",
        "entities": ["ORG", "PERSON", "GPE"],
        "model": "gpt-4o",
        "mock_response": load_output(
            "openai", "text", "custom_named_entity_recognition"
        )["original_response"]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_sentiment_analysis_params():
    params = {
        "text": "This is a great product!",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "text", "sentiment_analysis")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_language_detection_params():
    params = {
        "text": "This is an English sentence.",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "translation", "language_detection")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params


@pytest.fixture
def mocked_automatic_translation_params():
    params = {
        "source_language": "en",
        "target_language": "fr",
        "text": "Hello, how are you?",
        "model": "gpt-4o",
        "mock_response": load_output("openai", "translation", "automatic_translation")[
            "original_response"
        ]["choices"][0]["message"]["content"],
    }
    return params
