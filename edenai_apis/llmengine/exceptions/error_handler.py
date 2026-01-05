import litellm
from edenai_apis.utils.exception import ProviderException

# Mapping from LiteLLM exception types to OpenAI error types and codes
LITELLM_TO_OPENAI_ERROR_MAPPING = {
    litellm.AuthenticationError: {
        "type": "authentication_error",
        "code": "invalid_api_key",
    },
    litellm.BadRequestError: {
        "type": "invalid_request_error",
        "code": "invalid_request",
    },
    litellm.RateLimitError: {
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
    },
    litellm.NotFoundError: {
        "type": "invalid_request_error",
        "code": "model_not_found",
    },
    litellm.APIError: {
        "type": "api_error",
        "code": "api_error",
    },
    litellm.InternalServerError: {
        "type": "server_error",
        "code": "internal_server_error",
    },
    litellm.ContentPolicyViolationError: {
        "type": "invalid_request_error",
        "code": "content_policy_violation",
    },
    litellm.ContextWindowExceededError: {
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    },
    litellm.Timeout: {
        "type": "timeout_error",
        "code": "request_timeout",
    },
    litellm.ServiceUnavailableError: {
        "type": "server_error",
        "code": "service_unavailable",
    },
    litellm.APIConnectionError: {
        "type": "api_error",
        "code": "connection_error",
    },
    litellm.APIResponseValidationError: {
        "type": "api_error",
        "code": "invalid_response",
    },
    litellm.UnprocessableEntityError: {
        "type": "invalid_request_error",
        "code": "unprocessable_entity",
    },
    litellm.JSONSchemaValidationError: {
        "type": "invalid_request_error",
        "code": "json_schema_validation_error",
    },
    litellm.UnsupportedParamsError: {
        "type": "invalid_request_error",
        "code": "unsupported_params",
    },
}


def handle_litellm_exception(e: Exception) -> ProviderException:
    """
    Transform LiteLLM exceptions into OpenAI-compatible ProviderException.

    Args:
        e: LiteLLM exception to transform

    Returns:
        ProviderException with OpenAI-compatible fields including:
        - message: Cleaned error message
        - code: HTTP status code
        - error_type: OpenAI error type (e.g., "invalid_request_error")
        - error_code: OpenAI error code (e.g., "invalid_api_key")
        - param: Parameter that caused the error (if available)
        - llm_provider: Provider name from LiteLLM (if available)
    """
    default_status_codes = {
        litellm.APIError: 500,
        litellm.APIConnectionError: 503,
        litellm.APIResponseValidationError: 422,
        litellm.AuthenticationError: 401,
        litellm.BadRequestError: 400,
        litellm.NotFoundError: 404,
        litellm.RateLimitError: 429,
        litellm.ServiceUnavailableError: 503,
        litellm.ContentPolicyViolationError: 400,
        litellm.Timeout: 408,
        litellm.UnprocessableEntityError: 422,
        litellm.JSONSchemaValidationError: 400,
        litellm.UnsupportedParamsError: 400,
        litellm.ContextWindowExceededError: 400,
        litellm.InternalServerError: 500,
    }

    # Extract and clean message
    original_message = str(e)
    cleaned_message = original_message.replace("litellm.", "")

    for error_type in default_status_codes.keys():
        error_name = error_type.__name__
        cleaned_message = cleaned_message.replace(f"{error_name}: ", "")

    # Extract status code
    status_code = getattr(e, "status_code", None)
    if status_code is None:
        for error_type, code in default_status_codes.items():
            if isinstance(e, error_type):
                status_code = code
                break

    if status_code is None:
        status_code = 500

    # Extract OpenAI error type and code from mapping
    # Check most specific exceptions first (order matters due to inheritance)
    error_type = None
    error_code = None

    # Check specific subclasses first before their parent classes
    specific_exceptions = [
        litellm.ContextWindowExceededError,
        litellm.ContentPolicyViolationError,
        litellm.JSONSchemaValidationError,
        litellm.UnsupportedParamsError,
    ]

    for exception_class in specific_exceptions:
        if isinstance(e, exception_class):
            error_info = LITELLM_TO_OPENAI_ERROR_MAPPING[exception_class]
            error_type = error_info["type"]
            error_code = error_info["code"]
            break

    # If not matched yet, check remaining exceptions
    if error_type is None:
        for exception_class, error_info in LITELLM_TO_OPENAI_ERROR_MAPPING.items():
            if isinstance(e, exception_class):
                error_type = error_info["type"]
                error_code = error_info["code"]
                break

    # Extract additional fields from LiteLLM exception if available
    param = getattr(e, "param", None)
    llm_provider = getattr(e, "llm_provider", None)

    return ProviderException(
        message=cleaned_message,
        code=status_code,
        error_type=error_type,
        error_code=error_code,
        param=param,
        llm_provider=llm_provider,
    )
