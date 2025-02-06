from typing import Any, List
import pytest

from edenai_apis.llm_engine.tests.fixtures.mocked_response import (
    mocked_chat_params,
    mocked_multimodal_chat_params,
    mocked_summarize_params,
    mocked_topic_extraction_params,
    mocked_automatic_translation_params,
    mocked_code_generation_params,
    mocked_custom_classification_params,
    mocked_custom_ner_params,
    mocked_keyword_extraction_params,
    mocked_language_detection_params,
    mocked_named_entity_recognition_params,
    mocked_pii_params,
    mocked_sentiment_analysis_params,
    mocked_spell_check_params,
)
from edenai_apis.llm_engine.types.litellm_model import LiteLLMModel


@pytest.fixture
def llm_engine_instance_wo_model():
    params_wo_model = {
        "provider_name": "test_provider_name",
        "client_name": "litellm",
        "application_name": "test_application",
    }
    return params_wo_model


@pytest.fixture
def llm_engine_instance_w_model():
    params_w_model = {
        "provider_name": "openai",
        "client_name": "litellm",
        "application_name": "test_application",
    }
    return params_w_model


@pytest.fixture
def llm_engine_instance_wo_oai_model():
    params_wo_model = {
        "provider_name": "openai",
    }
    return params_wo_model


@pytest.fixture
def llm_engine_instance_w_unregestired_model():
    params_w_model = {
        "provider_name": "openai",
        "client_name": "litellm",
        "application_name": "test_application",
        "model_name": "test-inexisting-model",
    }
    return params_w_model


@pytest.fixture
def mocked_completion_params(model: str = "gpt-4o-mini"):
    params = {
        "text": "Hello",
        "client_name": "litellm",
        "model": model,
        "chatbot_global_action": "You are a helpful assistant.",
        "previous_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help you?"},
        ],
        "temperature": 0.2,
        "max_tokens": 100,
        "top_p": 0.5,
        "top_k": 50,
        "stop_sequences": ["\n\n"],
        "api_key": "Somerandomapikey",
        "stream": False,
        "mock_response": "Hey, this is the testing machine",
    }
    return params


@pytest.fixture
def mocked_embedding_params():
    params = {
        "client_name": "litellm",
        "model": "text-embedding-ada-002",
        "input": ["Embed this"],
        "api_key": "Somerandomapikey",
        "mock_response": [0, 0.2, 0.3, 0.4, 0.5],
    }
    return params


@pytest.fixture
def mocked_image_generation_params():
    params = {
        "client_name": "litellm",
        "api_key": "Somerandomapikey",
        "prompt": "some image to generate",
        "model": "dall-e-3",
        "n": 1,
        "size": "1024x1024",
        "mock_response": "url.net",
    }
    return params


@pytest.fixture
def mocked_completion_parametrized():
    params = {
        "text": "hey how are you ? ",
        "chatbot_global_action": "Act as an assistant",
        "previous_history": [],
        "temperature": 0,
        "max_tokens": 120,
        "model": "test-inexisting-model",
        "stream": False,
        "available_tools": None,
        "tool_choice": "auto",
        "tool_results": None,
        "mock_response": "Hey, this is the testing machine",
    }
    return params


@pytest.fixture
def unknown_models_to_litellm() -> List[LiteLLMModel]:
    test_model = {
        "test-inexisting-model": {
            "max_tokens": 131072,
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000001,
            "litellm_provider": "openai",  # I thing we need to use a valid existing provider
            "mode": "completion",
        }
    }
    return test_model
