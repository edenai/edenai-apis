from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from litellm import register_model
from llmengine.clients.litellm_client.litellm_client import LiteLLMCompletionClient
from llmengine.llm_engine import LLMEngine, StdLLMEngine

from edenai_apis.features.multimodal.chat.chat_dataclass import (
    ChatStreamResponse as ChatMultimodalStreamResponse,
)
from edenai_apis.features.multimodal.chat.chat_dataclass import (
    StreamChat as StreamChatMultimodal,
)
from edenai_apis.features.text.chat.chat_dataclass import ChatStreamResponse, StreamChat
from edenai_apis.llmengine.types.litellm_model import LiteLLMModel
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class TestLiteLLMClient:
    @pytest.mark.unit
    def test_client_lookup(self):
        from llmengine.clients import LLM_COMPLETION_CLIENTS

        assert "litellm" in LLM_COMPLETION_CLIENTS

    @pytest.mark.unit
    def test_llm_engine_instantiation(
        self, llm_engine_instance_wo_model, llm_engine_instance_w_model
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_model)
        assert llm_engine is not None
        llm_engine = None
        llm_engine = LLMEngine(**llm_engine_instance_w_model)
        assert llm_engine is not None

    @pytest.mark.integration
    def test_litellm_client_completion_call(
        self, llm_engine_instance_wo_oai_model, mocked_completion_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion_client.completion(**mocked_completion_params)
        assert response["choices"] is not None
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["choices"][0]["message"]["content"] is not None

    @pytest.mark.integration
    def test_litellm_client_embedding_call(
        self, llm_engine_instance_wo_oai_model, mocked_embedding_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.completion_client.embedding(**mocked_embedding_params)
        assert response is not None

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_llm_engine_multimodal_chat(
        self, llm_engine_instance_wo_oai_model, mocked_multimodal_chat_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.multimodal_chat(**mocked_multimodal_chat_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response.generated_text == "hey hey"
        assert response.usage

    @pytest.mark.integration
    def test_llm_engine_summarize(
        self, llm_engine_instance_wo_oai_model, mocked_summarize_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.summarize(**mocked_summarize_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response.result == "summarized document"
        assert response.usage

    @pytest.mark.integration
    def test_llm_engine_topic_extraction(
        self, llm_engine_instance_wo_oai_model, mocked_topic_extraction_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.topic_extraction(**mocked_topic_extraction_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.usage

        # assert isinstance(response.standardized_response, TopicExtractionDataClass)

    @pytest.mark.integration
    def test_llm_engine_sentiment_analysis(
        self, llm_engine_instance_wo_oai_model, mocked_sentiment_analysis_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.sentiment_analysis(**mocked_sentiment_analysis_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
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

    @pytest.mark.integration
    def test_llm_engine_pii(self, llm_engine_instance_wo_oai_model, mocked_pii_params):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.pii(**mocked_pii_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    @pytest.mark.integration
    def test_llm_engine_code_generation(
        self, llm_engine_instance_wo_oai_model, mocked_code_generation_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.code_generation(**mocked_code_generation_params)
        assert isinstance(response, ResponseType)
        assert response.original_response
        assert response.standardized_response
        assert response.usage

    @pytest.mark.integration
    def test_llm_engine_custom_classification(
        self, llm_engine_instance_wo_oai_model, mocked_custom_classification_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.custom_classification(
            **mocked_custom_classification_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    @pytest.mark.integration
    def test_llm_engine_custom_named_entity_recognition(
        self, llm_engine_instance_wo_oai_model, mocked_custom_ner_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.custom_named_entity_recognition(
            **mocked_custom_ner_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    @pytest.mark.integration
    def test_llm_engine_language_detection(
        self, llm_engine_instance_wo_oai_model, mocked_language_detection_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.language_detection(**mocked_language_detection_params)
        assert isinstance(response, ResponseType)
        assert response.original_response

    @pytest.mark.integration
    def test_llm_engine_automatic_translation(
        self, llm_engine_instance_wo_oai_model, mocked_automatic_translation_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.automatic_translation(
            **mocked_automatic_translation_params
        )
        assert isinstance(response, ResponseType)
        assert response.original_response

    @pytest.mark.integration
    def test_llm_engine_chat_stream(
        self, llm_engine_instance_wo_oai_model, mocked_chat_stream_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.chat(**mocked_chat_stream_params)
        assert isinstance(response, ResponseType)
        assert response.original_response is None
        assert isinstance(response.standardized_response, StreamChat)
        assert isinstance(response.standardized_response.stream, Iterator)
        chat_response = ""
        for chunk in response.standardized_response.stream:
            assert isinstance(chunk, ChatStreamResponse)
            chat_response += chunk.text
        assert chat_response == mocked_chat_stream_params["mock_response"]

    @pytest.mark.integration
    def test_llm_engine_chat_multimodal_stream(
        self, llm_engine_instance_wo_oai_model, mocked_multimodal_chat_stream_params
    ):
        llm_engine = LLMEngine(**llm_engine_instance_wo_oai_model)
        response = llm_engine.multimodal_chat(**mocked_multimodal_chat_stream_params)
        assert isinstance(response, ResponseType)
        assert response.original_response is None
        assert isinstance(response.standardized_response, StreamChatMultimodal)
        assert isinstance(response.standardized_response.stream, Iterator)
        chat_response = ""
        for chunk in response.standardized_response.stream:
            assert isinstance(chunk, ChatMultimodalStreamResponse)
            chat_response += chunk.text
        assert chat_response == mocked_multimodal_chat_stream_params["mock_response"]

    # class TestLLMClients:

    @pytest.mark.integration
    def test_unregisterd_model(
        self, llm_engine_instance_w_unregestired_model, mocked_completion_parametrized
    ):
        with pytest.raises(ProviderException):
            llm_engine = LLMEngine(**llm_engine_instance_w_unregestired_model)
            llm_engine.chat(**mocked_completion_parametrized)

    @pytest.mark.integration
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


class TestStdLLMEngine:

    @pytest.fixture
    def mock_execute_completion(self):
        with patch.object(StdLLMEngine, "_execute_completion") as mock_execute:
            mock_execute.return_value = {"response": "success"}
            yield mock_execute

    @pytest.fixture
    def mock_load_provider(self):
        with patch("llmengine.llm_engine.load_provider") as mock_provider_key:
            mock_provider_key.return_value = {
                "api_key": "test_key",
                "genai_api_key": "test_key",
            }
            yield mock_provider_key

    @pytest.mark.unit
    def test_map_provider(self, mapping_providers):
        for source, target in mapping_providers:
            assert StdLLMEngine.map_provider(source) == target

    @pytest.mark.unit
    def test_completion(self, mock_execute_completion, mock_load_provider):
        engine = StdLLMEngine()
        response = engine.completion(
            messages=[{"role": "user", "content": "Hello"}], provider="openai"
        )

        assert response == {"response": "success"}
        mock_execute_completion.assert_called_once()
        mock_load_provider.assert_called()

    @pytest.mark.unit
    def test_completion_google(self, mock_execute_completion):
        with patch("llmengine.llm_engine.load_provider") as mock_load_provider:
            mock_load_provider.return_value = (
                {"api_key": "test_key", "genai_api_key": "test_key", "project_id": ""},
                "",
            )
            engine = StdLLMEngine()

            response = engine.completion(messages=[], provider="vertex_ai")
            call_args, call_kwargs = mock_load_provider.call_args
            assert call_args[0].value == ProviderDataEnum.KEY.value
            assert call_kwargs == {
                "provider_name": "google",
                "location": True,
                "api_keys": None,
            }
            mock_execute_completion.assert_called_once()
            assert response is not None

    @pytest.mark.unit
    def test_completion_gemini(self, mock_execute_completion, mock_load_provider):
        engine = StdLLMEngine()

        response = engine.completion(messages=[], provider="gemini")
        call_args, call_kwargs = mock_load_provider.call_args
        assert call_args[0].value == ProviderDataEnum.KEY.value
        assert call_kwargs == {
            "provider_name": "google",
            "api_keys": None,
        }
        mock_execute_completion.assert_called_once()
        assert response is not None

    @pytest.mark.unit
    def test_completion_no_provider(self, mock_execute_completion, mock_load_provider):
        mock_execute_completion.return_value = MagicMock()
        mock_load_provider.return_value = {"api_key": "test_api_key"}
        engine = StdLLMEngine()

        response = engine.completion(messages=[])
        mock_load_provider.assert_not_called()
        mock_execute_completion.assert_called_once()
        assert response is not None

    @pytest.mark.integration
    def test_execute_completion(self):
        engine = StdLLMEngine()
        with patch.object(engine, "completion_client") as mock_completion_client:
            mock_completion_client.completion.return_value = {
                "text": "Generated response"
            }
            params = {"messages": [{"role": "user", "content": "Hello"}]}
            response = engine._execute_completion(params)

            assert response.text == "Generated response"
            mock_completion_client.completion.assert_called_once_with(**params)
