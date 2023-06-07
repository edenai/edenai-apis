from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Requested sentences exceed total sentences detected in text",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Input text not formatted correctly. Error Too short text, should be longer than \d+ characters",
    ],
}
