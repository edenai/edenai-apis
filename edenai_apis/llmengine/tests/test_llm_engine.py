from typing import List
import pytest
from edenai_apis.llmengine.types.litellm_model import LiteLLMModel
from edenai_apis.utils.exception import ProviderException
from llmengine.llm_engine import LLMEngine
from llmengine.clients.litellm_client.litellm_client import LiteLLMCompletionClient
class TestLLMEngine:

    def test_client_lookup(self):
        from llmengine.clients import LLM_COMPLETION_CLIENTS
        assert "litellm" in LLM_COMPLETION_CLIENTS

    def test_llm_engine_instantiation(self, llm_engine_instance_wo_model, llm_engine_instance_w_model):
        llm_engine = LLMEngine(**llm_engine_instance_wo_model)
        assert llm_engine is not None
        llm_engine = None
        llm_engine = LLMEngine(**llm_engine_instance_w_model)
        assert llm_engine is not None

    def test_llm_engine_completion_call(self, llm_engine_instance_wo_oai_model, mocked_completion_params):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.chat(**mocked_completion_params)
        assert response.original_response["choices"] is not None
        assert response.original_response["choices"][0]["finish_reason"] == "stop"
        assert response.original_response["choices"][0]["message"]["content"] is not None


    def test_llm_engine_embedding_call(self, llm_engine_instance_wo_oai_model, mocked_embedding_params):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.embedding(**mocked_embedding_params)
        print(response)
        assert response is not None

class TestLLMClients:

    def test_unregisterd_model(self, llm_engine_instance_w_unregestired_model, mocked_completion_parametrized):
        with pytest.raises(ProviderException):
            llm_engine = LLMEngine(**llm_engine_instance_w_unregestired_model)
            llm_engine.chat(**mocked_completion_parametrized)
    
    def test_register_and_use_model(self, llm_engine_instance_w_model, mocked_completion_parametrized, unknown_models_to_litellm: List[LiteLLMModel]):
        ## 
        LiteLLMCompletionClient.register_new_models(unknown_models_to_litellm)
        ##
        llm_engine = LLMEngine(**llm_engine_instance_w_model)
        response = llm_engine.chat(**mocked_completion_parametrized)
        assert response.original_response is not None
        assert response.original_response["choices"][0] is not None
        assert response.original_response["choices"][0]["finish_reason"] == "stop"
        assert response.original_response["choices"][0]["message"]["content"] is not None