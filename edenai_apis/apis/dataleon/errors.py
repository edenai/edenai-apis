from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputFileFormatError,
    ProviderTimeoutError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileFormatError: [
        r"File type not managed",
        r"Provider dataleon doesn't support file type: \w+ for this feature. Supported mimetypes are \w+",
    ],
    ProviderInternalServerError: [
        r"500 Internal Server Error",
        r"\w+ 502 Bad Gateway \w+",
        r"\w+ 503 Service Temporarily Unavailable \w+",
    ],
    ProviderTimeoutError: [r"\w+ 504 Gateway Time-out \w+"],
}
