from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputTextLengthError,
    ProviderAuthorizationError,
    ProviderNotFoundError,
    ProviderInvalidInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"Maximum 500 characters.",
    ],
    ProviderInternalServerError: [
        r"Internal server error",
    ],
    ProviderAuthorizationError: [r"Invalid API Key"],
    ProviderNotFoundError: [
        r"Unavailable voice skin.",
    ],
    ProviderInvalidInputError: [
        r"Speaker \w+ for language \w+ is not available",
        r"Wrong voice id",
    ],
}
