import pytest
from llm_engine.llm_engine import LLMEngine

class TestLLMEngine:

    def test_client_lookup(self):
        from llm_engine.clients import LLM_COMPLETION_CLIENTS
        assert "litellm" in LLM_COMPLETION_CLIENTS

    def test_llm_engine_instantiation(self, llm_engine_instance_wo_model, llm_engine_instance_w_model):
        llm_engine = LLMEngine(**llm_engine_instance_wo_model)
        assert llm_engine is not None
        llm_engine = None
        llm_engine = LLMEngine(**llm_engine_instance_w_model)
        assert llm_engine is not None

    def test_llm_engine_completion_call(self, llm_engine_instance_wo_oai_model, mocked_completion_params):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion(**mocked_completion_params)
        assert response.choices is not None
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].message.content is not None
        