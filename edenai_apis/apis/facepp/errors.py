from edenai_apis.utils.exception import (
    ProviderAuthorizationError,
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputPayloadSize,
    ProviderLimitationError,
    ProviderNotFoundError,
    ProviderPermissionError,
    ProviderTimeoutError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"INVALID_IMAGE_URL",
        r"MISSING_ARGUMENTS",
        r"BAD_ARGUMENTS",
        r"COEXISTENCE_ARGUMENTS",
        r"TOO_MANY_FACE_ATTRIBUTES",
        r"INVALID_FACESET_TOKEN",
        r"INVALID_OUTER_ID",
        r"NEW_OUTER_ID_EXIST",
        r"INVALID_FACE_TOKENS_SIZE",
        r"FACESET_EXIST",
    ],
    ProviderInvalidInputFileFormatError: [
        r"IMAGE_ERROR_UNSUPPORTED_FORMAT",
    ],
    ProviderInvalidInputFileSizeError: [
        r"INVALID_IMAGE_SIZE",
        r"IMAGE_FILE_TOO_LARGE",
    ],
    ProviderNotFoundError: [
        r"API_NOT_FOUND",
    ],
    ProviderInternalServerError: [
        r"INTERNAL_ERROR",
    ],
    ProviderInvalidInputPayloadSize: [
        r"Request Entity Too Large"
    ],
    ProviderPermissionError: [
        r"INSUFFICIENT_PERMISSION",
        r"AUTHORIZATION_ERROR",
    ],
    ProviderAuthorizationError: [
        r"AUTHENTICATION_ERROR",
    ],
    ProviderLimitationError: [
        r"CONCURRENCY_LIMIT_EXCEEDED",
        r"FACESET_QUOTA_EXCEEDED",
    ],
    ProviderTimeoutError: [
        r"IMAGE_DOWNLOAD_TIMEOUT",
    ],
}
