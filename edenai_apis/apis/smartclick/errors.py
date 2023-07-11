from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInternalServerError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"invalid url or image",
    ],
    ProviderInternalServerError: [
        r"Service Temporarily Unavailable",
        r"Internal Server Error",
    ],
}
