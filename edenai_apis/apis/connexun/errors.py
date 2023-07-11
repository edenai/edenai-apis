from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
    ProviderAuthorizationError,
    ProviderParsingError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Requested sentences exceed total sentences detected in text",
        r"Invalid language format for: \w+.",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Input text not formatted correctly. Error Too \w+ text, should be \w+ than \d+ characters",
    ],
    ProviderAuthorizationError: [
        r"Forbidden",
    ],
    ProviderParsingError: [r"Provider has not found a sentiment of the text."],
}
