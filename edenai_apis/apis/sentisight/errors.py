from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderTimeoutError,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"HTTP 404 Not Found",
        r"invalid url or image",
        r"Provider does not support selected language: \w+",
    ],
    ProviderInternalServerError: [
        r"Internal Server Error",
        r"Unknown error",
    ],
    ProviderTimeoutError: [
        r"504 Gateway Time-out",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Image size \(\d+ pixels\) exceeds limit of \d+ pixels, could be decompression bomb DOS attack."
    ],
    ProviderInvalidInputFileFormatError: [
        r"Provider sentisight doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ],
}
