from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
    ProviderInternalServerError,
    ProviderMissingInputError,
    ProviderLimitationError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"The text must not be greater than 4000 characters.",
    ],
    ProviderInvalidInputError: [
        r"The language defined in \w+ locale is not yet supported",
        r"Invalid language format for: \w+."
    ],
    ProviderInternalServerError : [
        r"Server Error"
    ],
    ProviderMissingInputError : [
        r"The \w+ locale field is required."
    ],
    ProviderLimitationError : [
        r"Too Many Attempts."
    ]

}
