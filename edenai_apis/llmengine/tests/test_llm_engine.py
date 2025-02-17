from typing import List
import pytest
from edenai_apis.llmengine.types.litellm_model import LiteLLMModel
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from litellm import register_model

from llmengine.llm_engine import LLMEngine
from llmengine.clients.litellm_client.litellm_client import LiteLLMCompletionClient


class TestLiteLLMClient:
    def test_client_lookup(self):
        from llmengine.clients import LLM_COMPLETION_CLIENTS

        assert "litellm" in LLM_COMPLETION_CLIENTS

    def test_llm_engine_instantiation(
        self, llm_engine_instance_wo_model, llm_engine_instance_w_model
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_model)
        assert llm_engine is not None
        llm_engine = None
        llm_engine = LLMEngine(**llm_engine_instance_w_model)
        assert llm_engine is not None

    def test_litellm_client_completion_call(
        self, llm_engine_instance_wo_oai_model, mocked_completion_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion_client.completion(**mocked_completion_params)
        assert response["choices"] is not None
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["choices"][0]["message"]["content"] is not None

    def test_litellm_client_embedding_call(
        self, llm_engine_instance_wo_oai_model, mocked_embedding_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion_client.embedding(**mocked_embedding_params)
        assert response is not None

    def test_litellm_client_image_generation_call(
        self, llm_engine_instance_wo_oai_model, mocked_image_generation_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion_client.image_generation(
            **mocked_image_generation_params
        )
        assert response is not None
        assert response["data"] is not None
        assert response["data"][0]["url"] == "url.net"


class TestLLMEngine:

    def test_llm_engine_chat(
        self, llm_engine_instance_wo_oai_model, mocked_chat_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.chat(**mocked_chat_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert (
            response.standardized_response.generated_text
            == "Hey, this is the testing machine"
        )
        assert response.usage

    def test_llm_engine_multimodal_chat(
        self, llm_engine_instance_wo_oai_model, mocked_multimodal_chat_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.multimodal_chat(**mocked_multimodal_chat_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response.generated_text == "hey hey"
        assert response.usage

    def test_llm_engine_summarize(
        self, llm_engine_instance_wo_oai_model, mocked_summarize_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.summarize(**mocked_summarize_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response.result == "summarized document"
        assert response.usage

    def test_llm_engine_topic_extraction(
        self, llm_engine_instance_wo_oai_model, mocked_topic_extraction_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.topic_extraction(**mocked_topic_extraction_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.usage

        # assert isinstance(response.standardized_response, TopicExtractionDataClass)

    def test_llm_engine_sentiment_analysis(
        self, llm_engine_instance_wo_oai_model, mocked_sentiment_analysis_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.sentiment_analysis(**mocked_sentiment_analysis_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    def test_llm_engine_keyword_extraction(
        self, llm_engine_instance_wo_oai_model, mocked_keyword_extraction_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.keyword_extraction(**mocked_keyword_extraction_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

        # assert isinstance(response.standardized_response, KeywordExtractionDataClass)

    def test_llm_engine_spell_check(
        self, llm_engine_instance_wo_oai_model, mocked_spell_check_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.spell_check(**mocked_spell_check_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage
        # assert isinstance(response.standardized_response, SpellCheckDataClass)

    def test_llm_engine_named_entity_recognition(
        self, llm_engine_instance_wo_oai_model, mocked_named_entity_recognition_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.named_entity_recognition(
            **mocked_named_entity_recognition_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    def test_llm_engine_pii(self, llm_engine_instance_wo_oai_model, mocked_pii_params):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.pii(**mocked_pii_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    def test_llm_engine_code_generation(
        self, llm_engine_instance_wo_oai_model, mocked_code_generation_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.code_generation(**mocked_code_generation_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    def test_llm_engine_custom_classification(
        self, llm_engine_instance_wo_oai_model, mocked_custom_classification_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.custom_classification(
            **mocked_custom_classification_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    def test_llm_engine_custom_named_entity_recognition(
        self, llm_engine_instance_wo_oai_model, mocked_custom_ner_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.custom_named_entity_recognition(
            **mocked_custom_ner_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    def test_llm_engine_language_detection(
        self, llm_engine_instance_wo_oai_model, mocked_language_detection_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.language_detection(**mocked_language_detection_params)
        assert isinstance(response, ResponseType)
        assert response.original_response

    def test_llm_engine_automatic_translation(
        self, llm_engine_instance_wo_oai_model, mocked_automatic_translation_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.automatic_translation(
            **mocked_automatic_translation_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    # class TestLLMClients:

    def test_unregisterd_model(
        self, llm_engine_instance_w_unregestired_model, mocked_completion_parametrized
    ):
        with pytest.raises(ProviderException):
            llm_engine = LLMEngine(**llm_engine_instance_w_unregestired_model)
            llm_engine.chat(**mocked_completion_parametrized)

    def test_register_and_use_model(
        self,
        llm_engine_instance_wo_oai_model,
        mocked_completion_parametrized,
        unknown_models_to_litellm,
    ):
        register_model(model_cost=unknown_models_to_litellm)
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.chat(**mocked_completion_parametrized)
        assert response.original_response is not None
        assert response.original_response["choices"][0] is not None
        assert response.original_response["choices"][0]["finish_reason"] == "stop"
        assert (
            response.original_response["choices"][0]["message"]["content"] is not None
        )
