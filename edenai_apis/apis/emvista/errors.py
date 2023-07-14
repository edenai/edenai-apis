from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputTextLengthError,
    ProviderInvalidInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Analyse error",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Input text is too long",
    ],
    ProviderInvalidInputError: [
        r"Invalid language format for: \w+.",
        r"This provider doesn't auto-detect languages, please provide a valid language",
    ],
}
