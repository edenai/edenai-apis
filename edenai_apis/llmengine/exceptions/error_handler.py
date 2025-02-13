import litellm
from edenai_apis.utils.exception import ProviderException


def handle_litellm_exception(e: Exception) -> ProviderException:
    """
    Transform LiteLLM exceptions into ProviderException
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

    original_message = str(e)
    cleaned_message = original_message.replace("litellm.", "")

    for error_type in default_status_codes.keys():
        error_name = error_type.__name__
        cleaned_message = cleaned_message.replace(f"{error_name}: ", "")

    status_code = getattr(e, "status_code", None)
    if status_code is None:
        for error_type, code in default_status_codes.items():
            if isinstance(e, error_type):
                status_code = code
                break

    if status_code is None:
        status_code = 500

    return ProviderException(message=cleaned_message, code=status_code)
