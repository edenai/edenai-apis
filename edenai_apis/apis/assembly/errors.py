from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
    ProviderAuthorizationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"`.*` list may not contain digits",
        r".* was included in the word boost list but contains unsupported characters",
    ],
    ProviderInvalidInputFileFormatError: [
        r"File extension not supported. Use one of the following extensions: \w+"
    ],
    ProviderAuthorizationError: [
        r"Your current account balance is negative. Please top up to continue using the API."
    ],
}
