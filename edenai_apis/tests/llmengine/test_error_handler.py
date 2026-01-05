"""
Tests for OpenAI-compatible error handling in LLM engine.

This module tests the conversion of LiteLLM exceptions to OpenAI-compatible
error format through the ProviderException class.
"""
from unittest.mock import Mock

import pytest
import litellm
from edenai_apis.llmengine.exceptions.error_handler import (
    handle_litellm_exception,
    LITELLM_TO_OPENAI_ERROR_MAPPING,
)
from edenai_apis.utils.exception import ProviderException


class TestProviderExceptionOpenAIFormat:
    """Test ProviderException to_openai_format method."""

    @pytest.mark.unit
    def test_basic_provider_exception_backward_compatibility(self):
        """Test that ProviderException maintains backward compatibility."""
        exc = ProviderException(message="Test error", code=400)
        assert str(exc) == "Test error"
        assert exc.status_code == 400
        assert exc.code == 400

    @pytest.mark.unit
    def test_provider_exception_with_openai_fields(self):
        """Test ProviderException with OpenAI-compatible fields."""
        exc = ProviderException(
            message="Invalid API key",
            code=401,
            error_type="authentication_error",
            error_code="invalid_api_key",
            param=None,
            llm_provider="openai",
        )
        assert exc.error_type == "authentication_error"
        assert exc.error_code == "invalid_api_key"
        assert exc.param is None
        assert exc.llm_provider == "openai"

    @pytest.mark.unit
    def test_to_openai_format_with_all_fields(self):
        """Test to_openai_format with all fields populated."""
        exc = ProviderException(
            message="Invalid request parameter",
            code=400,
            error_type="invalid_request_error",
            error_code="invalid_request",
            param="temperature",
            llm_provider="openai",
        )
        result = exc.to_openai_format()

        assert "error" in result
        assert result["error"]["message"] == "Invalid request parameter"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["code"] == "invalid_request"
        assert result["error"]["param"] == "temperature"

    @pytest.mark.unit
    def test_to_openai_format_with_minimal_fields(self):
        """Test to_openai_format with minimal fields (backward compatibility)."""
        exc = ProviderException(message="Something went wrong", code=500)
        result = exc.to_openai_format()

        assert "error" in result
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "server_error"  # Inferred from code
        assert result["error"]["param"] is None
        assert "code" not in result["error"]  # No error_code provided

    @pytest.mark.unit
    def test_to_openai_format_type_inference(self):
        """Test that error type is correctly inferred from HTTP status code."""
        test_cases = [
            (401, "authentication_error"),
            (429, "rate_limit_error"),
            (500, "server_error"),
            (503, "server_error"),
            (400, "invalid_request_error"),
            (404, "invalid_request_error"),
        ]

        for status_code, expected_type in test_cases:
            exc = ProviderException(message="Test error", code=status_code)
            result = exc.to_openai_format()
            assert result["error"]["type"] == expected_type

    @pytest.mark.unit
    def test_to_openai_format_without_code(self):
        """Test to_openai_format when no status code is provided."""
        exc = ProviderException(message="Generic error")
        result = exc.to_openai_format()

        assert result["error"]["type"] == "api_error"  # Fallback
        assert result["error"]["message"] == "Generic error"


class TestHandleLiteLLMException:
    """Test handle_litellm_exception function."""

    @pytest.mark.unit
    def test_authentication_error_mapping(self):
        """Test AuthenticationError is correctly mapped."""
        litellm_exc = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4",
            response=Mock(),
        )
        litellm_exc.status_code = 401

        provider_exc = handle_litellm_exception(litellm_exc)

        assert isinstance(provider_exc, ProviderException)
        assert provider_exc.status_code == 401
        assert provider_exc.error_type == "authentication_error"
        assert provider_exc.error_code == "invalid_api_key"

    @pytest.mark.unit
    def test_rate_limit_error_mapping(self):
        """Test RateLimitError is correctly mapped."""
        # Create a proper mock response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}

        litellm_exc = litellm.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4",
            response=mock_response,
        )

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.status_code == 429
        assert provider_exc.error_type == "rate_limit_error"
        assert provider_exc.error_code == "rate_limit_exceeded"

    @pytest.mark.unit
    def test_bad_request_error_mapping(self):
        """Test BadRequestError is correctly mapped."""
        litellm_exc = litellm.BadRequestError(
            message="Invalid parameter value",
            llm_provider="openai",
            model="gpt-4",
            response=Mock(),
        )
        litellm_exc.status_code = 400

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.status_code == 400
        assert provider_exc.error_type == "invalid_request_error"
        assert provider_exc.error_code == "invalid_request"

    @pytest.mark.unit
    def test_context_window_exceeded_error_mapping(self):
        """Test ContextWindowExceededError is correctly mapped."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        litellm_exc = litellm.ContextWindowExceededError(
            message="Context length exceeded",
            llm_provider="openai",
            model="gpt-4",
            response=mock_response,
        )

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.status_code == 400
        assert provider_exc.error_type == "invalid_request_error"
        assert provider_exc.error_code == "context_length_exceeded"

    @pytest.mark.unit
    def test_internal_server_error_mapping(self):
        """Test InternalServerError is correctly mapped."""
        litellm_exc = litellm.InternalServerError(
            message="Internal server error",
            llm_provider="openai",
            model="gpt-4",
            response=Mock(),
        )
        litellm_exc.status_code = 500

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.status_code == 500
        assert provider_exc.error_type == "server_error"
        assert provider_exc.error_code == "internal_server_error"

    @pytest.mark.unit
    def test_timeout_error_mapping(self):
        """Test Timeout is correctly mapped."""
        litellm_exc = litellm.Timeout(
            message="Request timeout",
            llm_provider="openai",
            model="gpt-4",
        )
        litellm_exc.status_code = 408

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.status_code == 408
        assert provider_exc.error_type == "timeout_error"
        assert provider_exc.error_code == "request_timeout"

    @pytest.mark.unit
    def test_message_cleaning(self):
        """Test that error messages are properly cleaned."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.headers = {}

        litellm_exc = litellm.BadRequestError(
            message="BadRequestError: litellm.BadRequestError: Invalid request",
            llm_provider="openai",
            model="gpt-4",
            response=mock_response,
        )

        provider_exc = handle_litellm_exception(litellm_exc)

        # Should have "litellm." and "BadRequestError: " removed
        assert "litellm." not in str(provider_exc)
        assert "Invalid request" in str(provider_exc)

    @pytest.mark.unit
    def test_status_code_fallback(self):
        """Test that status code falls back correctly when not provided."""
        litellm_exc = litellm.BadRequestError(
            message="Error without status code",
            llm_provider="openai",
            model="gpt-4",
            response=Mock(),
        )
        # Don't set status_code attribute

        provider_exc = handle_litellm_exception(litellm_exc)

        # Should default to 400 for BadRequestError
        assert provider_exc.status_code == 400

    @pytest.mark.unit
    def test_llm_provider_extraction(self):
        """Test that llm_provider is extracted from LiteLLM exception."""
        litellm_exc = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="anthropic",
            model="claude-3",
            response=Mock(),
        )

        provider_exc = handle_litellm_exception(litellm_exc)

        assert provider_exc.llm_provider == "anthropic"

    @pytest.mark.unit
    def test_all_exception_types_have_mapping(self):
        """Test that all exception types in default_status_codes have OpenAI mapping."""
        exception_types = [
            litellm.APIError,
            litellm.APIConnectionError,
            litellm.APIResponseValidationError,
            litellm.AuthenticationError,
            litellm.BadRequestError,
            litellm.NotFoundError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.ContentPolicyViolationError,
            litellm.Timeout,
            litellm.UnprocessableEntityError,
            litellm.JSONSchemaValidationError,
            litellm.UnsupportedParamsError,
            litellm.ContextWindowExceededError,
            litellm.InternalServerError,
        ]

        for exc_type in exception_types:
            assert (
                exc_type in LITELLM_TO_OPENAI_ERROR_MAPPING
            ), f"{exc_type.__name__} not in mapping"


class TestEndToEndErrorFormat:
    """Test end-to-end error format generation."""

    @pytest.mark.unit
    def test_complete_error_flow(self):
        """Test complete flow from LiteLLM exception to OpenAI format."""
        # Simulate a LiteLLM exception
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}

        litellm_exc = litellm.RateLimitError(
            message="Rate limit exceeded for requests",
            llm_provider="openai",
            model="gpt-4",
            response=mock_response,
        )
        litellm_exc.param = None

        # Handle the exception
        provider_exc = handle_litellm_exception(litellm_exc)

        # Convert to OpenAI format
        openai_format = provider_exc.to_openai_format()

        # Verify the complete structure
        assert openai_format == {
            "error": {
                "message": "Rate limit exceeded for requests",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        }

    @pytest.mark.unit
    def test_backward_compatible_error_flow(self):
        """Test that old code creating ProviderException still works."""
        # Old way of creating ProviderException
        provider_exc = ProviderException(message="Old style error", code=400)

        # Should still be able to convert to OpenAI format
        openai_format = provider_exc.to_openai_format()

        assert "error" in openai_format
        assert openai_format["error"]["message"] == "Old style error"
        assert openai_format["error"]["type"] == "invalid_request_error"
        assert openai_format["error"]["param"] is None
