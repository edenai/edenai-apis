import pytest

@pytest.fixture
def llm_engine_instance_wo_model():
    params_wo_model = {
        "provider_name": "test_provider_name",
        "client_name": "litellm",
        "application_name": "test_application"
    }
    return params_wo_model

@pytest.fixture
def llm_engine_instance_w_model():
    params_w_model = {
        "provider_name": "test_provider_name",
        "client_name": "litellm",
        "application_name": "test_application",
        "model_name": "test_model"
    }
    return params_w_model

@pytest.fixture
def llm_engine_instance_wo_oai_model():
    params_wo_model = {
        "provider_name": "openai",
        "client_name": "litellm",
        "application_name": "test_application"
    }
    return params_wo_model

@pytest.fixture
def mocked_completion_params():
    params = {
        "client_name": "litellm",
        "model": "gpt-4o-mini",
        "messages": [{"role":"user", "content":"Hello"}],
        "temperature": 0.2,
        "max_tokens": 100,
        "top_p": 0.5,
        "top_k": 50,
        "stop_sequences": ["\n\n"],
        "api_key": "Somerandomapikey",
        "mock_response": "Hey, this is the testing machine"
    }
    return params


@pytest.fixture
def mocked_embedding_params():
    params = {
        "client_name": "litellm",
        "model": "text-embedding-ada-002",
        "input": ["Embed this"],
        "api_key": "Somerandomapikey",
        "mock_response": [0, 0.2, 0.3, 0.4, 0.5]
    }
    return params