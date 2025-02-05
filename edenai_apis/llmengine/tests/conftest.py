from typing import Any, List
import pytest

from edenai_apis.llmengine.types.litellm_model import LiteLLMModel

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
        "provider_name": "openai",
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
def llm_engine_instance_w_unregestired_model():
    params_w_model = {
        "provider_name": "openai",
        "client_name": "litellm",
        "application_name": "test_application",
        "model_name": "test-inexisting-model"
    }
    return params_w_model

@pytest.fixture
def mocked_completion_params(model:str = "gpt-4o-mini"):
    params = {
        "text": "Hello",
        "client_name": "litellm",
        "model": model,
        "chatbot_global_action": "You are a helpful assistant.",
        "previous_history": [{"role":"user", "content":"Hello"}, {"role":"assistant", "content":"How can I help you?"}],
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

@pytest.fixture
def mocked_completion_parametrized(mocked_completion_params: dict[str, Any]):
    params = mocked_completion_params
    params["model"] = "test-inexisting-model"
    return params

@pytest.fixture
def unknown_models_to_litellm() -> List[LiteLLMModel]:
    test_model = LiteLLMModel.model_validate({
        "model_name": "test-inexisting-model",
        "model_configuration": {
            "max_tokens": 131072, 
            "input_cost_per_token": 0.000001, 
            "output_cost_per_token": 0.000001, 
            "litellm_provider": "openai", # I thing we need to use a valid existing provider
            "mode": "completion"
        }
    })
    return [test_model]