from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderNotFoundError,
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
}
