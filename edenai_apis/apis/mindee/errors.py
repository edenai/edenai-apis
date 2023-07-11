from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderNotFoundError,
    ProviderInvalidInputFileFormatError,
    ProviderLimitationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"An error occurred",
    ],
    ProviderInvalidInputError: [
        r"Invalid fields in form",
    ],
    ProviderNotFoundError: [
        r"Resource not found",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Provider mindee doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ],
    ProviderLimitationError: [r"Too many requests"],
}
