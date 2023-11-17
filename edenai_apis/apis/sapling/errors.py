# NOTE: error messages should be regex patterns
from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderParsingError,
    ProviderInvalidInputError,
    ProviderAuthorizationError,
)


ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [r"Internal server error"],
    ProviderParsingError: [r"Response malformed"],
    ProviderInvalidInputError: [r"400 Bad Request: {.}"],
    ProviderAuthorizationError: [r"Invalid api key"],
}
